#!/usr/bin/env python3
"""Wire contract between this client and the StarVLA policy server.

Pure numpy: no ROS, no OpenCV, so the mapping that decides which numbers reach
the arms can be tested off the robot.

Both directions carry **16 raw dims per frame**, laid out per arm as
``xyz(3) + quaternion(4) + gripper(1)``, left arm first::

    0:3   left position      8:11  right position
    3:7   left quaternion   11:15  right quaternion
    7     left gripper      15     right gripper

Quaternions on this wire are **xyzw**, the robot's order. The model works in
wxyz and in 20 rotation-6D dims; the server owns both conversions, so nothing
here has to know about them. See ``examples/Xtrainer2/README.md`` in the
StarVLA repository.
"""

import numpy as np

#: Camera order the model was trained on. Position in the list is meaningful:
#: the server maps entry i to its i-th view.
CAMERA_ORDER = ("cam_high", "cam_left_wrist", "cam_right_wrist")

#: Training instruction. Florence was trained on English only; the Chinese
#: source string will not reproduce the training distribution.
DEFAULT_INSTRUCTION = "Pick up the phone, place it in the box, and close the lid."

IMAGE_SIZE = (224, 224)
STATE_DIM = 16
ACTION_DIM = 16
POSE_DIM = 7

LEFT_POSE = slice(0, 7)
LEFT_GRIPPER = 7
RIGHT_POSE = slice(8, 15)
RIGHT_GRIPPER = 15

#: Substrings used to recognise each camera in the ROS chain-image frame ids.
_CAMERA_HINTS = {
    "cam_high": ("high", "head"),
    "cam_left_wrist": ("left",),
    "cam_right_wrist": ("right",),
}


def _as_pose(name, pose):
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    if pose.shape[0] != POSE_DIM:
        raise ValueError(
            f"{name} must be a {POSE_DIM}-D pose [x,y,z,qx,qy,qz,qw]; got shape {pose.shape}. "
            "multibody_pose carries the end-effector pose; multibody_state carries joint angles."
        )
    return pose


def _as_gripper(name, gripper):
    gripper = np.asarray(gripper, dtype=np.float32).reshape(-1)
    if gripper.shape[0] != 1:
        raise ValueError(f"{name} must be a single value; got shape {gripper.shape}")
    return gripper


def build_state(left_pose, left_gripper, right_pose, right_gripper):
    """Assemble the 16-D observation the server expects.

    Args:
        left_pose / right_pose: ``[x, y, z, qx, qy, qz, qw]`` end-effector pose
            in the robot torso frame (``ComponentObservation.multibody_pose``).
        left_gripper / right_gripper: continuous hand opening, not binarized.
    """
    state = np.concatenate(
        [
            _as_pose("left_pose", left_pose),
            _as_gripper("left_gripper", left_gripper),
            _as_pose("right_pose", right_pose),
            _as_gripper("right_gripper", right_gripper),
        ]
    )
    assert state.shape == (STATE_DIM,), state.shape
    return state


def split_action(action):
    """Split one 16-D server action into ``(left pose, left grip, right pose, right grip)``."""
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] != ACTION_DIM:
        raise ValueError(f"expected a {ACTION_DIM}-D action; got shape {action.shape}")
    return (
        action[LEFT_POSE],
        float(action[LEFT_GRIPPER]),
        action[RIGHT_POSE],
        float(action[RIGHT_GRIPPER]),
    )


def resolve_camera_indices(frame_ids, overrides=None):
    """Map each trained camera to an index into the robot's chain-image list.

    Ordering of that list is publisher-dependent, and a silent swap of the two
    wrist views is as damaging as a wrong quaternion, so this matches on frame
    id and refuses to guess.

    Args:
        frame_ids: frame id of every chain image, in the order received.
        overrides: ``{camera_name: frame_id_or_index}`` from the operator.

    Returns:
        ``{camera_name: index}`` for every entry of :data:`CAMERA_ORDER`.
    """
    overrides = overrides or {}
    frame_ids = list(frame_ids)
    resolved = {}

    for camera in CAMERA_ORDER:
        override = overrides.get(camera)
        if override is None:
            continue
        if isinstance(override, int) or (isinstance(override, str) and override.isdigit()):
            index = int(override)
            if not 0 <= index < len(frame_ids):
                raise ValueError(f"{camera}: index {index} outside 0..{len(frame_ids) - 1}")
        elif override in frame_ids:
            index = frame_ids.index(override)
        else:
            raise ValueError(f"{camera}: frame id {override!r} not among {frame_ids}")
        resolved[camera] = index

    for camera in CAMERA_ORDER:
        if camera in resolved:
            continue
        hints = _CAMERA_HINTS[camera]
        matches = [
            index
            for index, frame_id in enumerate(frame_ids)
            if any(hint in str(frame_id).lower() for hint in hints)
        ]
        matches = [index for index in matches if index not in resolved.values()]
        if len(matches) != 1:
            raise ValueError(
                f"cannot identify {camera} among chain images {frame_ids} "
                f"(hints {hints}, matched {matches}). "
                f"Pass --{camera.replace('_', '-')} <frame_id|index>."
            )
        resolved[camera] = matches[0]

    if len(set(resolved.values())) != len(CAMERA_ORDER):
        raise ValueError(f"cameras resolved to duplicate chain images: {resolved}")
    return resolved


def check_server_metadata(metadata):
    """Fail fast when the server was not trained for this robot layout.

    The handshake states the widths, camera order and quaternion order the
    checkpoint expects. A mismatch here means the arms would be commanded with
    numbers that mean something else.
    """
    problems = []

    camera_order = [str(key).split(".", 1)[-1] for key in metadata.get("camera_order", [])]
    if camera_order and tuple(camera_order) != CAMERA_ORDER:
        problems.append(f"camera order {camera_order} != {list(CAMERA_ORDER)}")

    if metadata.get("robot_quaternion_format", "xyzw") != "xyzw":
        problems.append(f"server expects quaternions in {metadata['robot_quaternion_format']}")

    for field, expected in (("state_dim", STATE_DIM), ("action_dim", ACTION_DIM)):
        actual = metadata.get(field)
        if actual is not None and int(actual) != expected:
            problems.append(f"{field} {actual} != {expected}")

    if not metadata.get("server_normalizes_state", False):
        problems.append(
            "server does not normalize state; this client only sends raw robot state"
        )

    image_format = metadata.get("image_format")
    if image_format not in (None, "HWC_RGB_uint8"):
        problems.append(f"server wants images as {image_format}")

    if problems:
        raise RuntimeError(
            "policy server contract mismatch:\n  - " + "\n  - ".join(problems)
        )
