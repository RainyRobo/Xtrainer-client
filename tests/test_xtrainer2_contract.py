"""Guards for the numbers that reach the arms.

Runs without ROS: it exercises only ``vla/xtrainer2_contract.py``.
"""

import numpy as np
import pytest

from vla.xtrainer2_contract import (
    ACTION_DIM,
    CAMERA_ORDER,
    STATE_DIM,
    build_state,
    check_server_metadata,
    resolve_camera_indices,
    split_action,
)


def _metadata(**overrides):
    metadata = {
        "camera_order": ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"],
        "state_dim": 16,
        "action_dim": 16,
        "robot_quaternion_format": "xyzw",
        "server_normalizes_state": True,
        "image_format": "HWC_RGB_uint8",
        "action_chunk_size": 16,
    }
    metadata.update(overrides)
    return metadata


def test_state_layout_is_pose_then_gripper_per_arm():
    left_pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
    right_pose = np.array([0.4, 0.5, 0.6, 0.0, 0.0, 1.0, 0.0])
    state = build_state(left_pose, [0.7], right_pose, [0.8])

    assert state.shape == (STATE_DIM,)
    assert np.allclose(state[0:7], left_pose)
    assert state[7] == pytest.approx(0.7)
    assert np.allclose(state[8:15], right_pose)
    assert state[15] == pytest.approx(0.8)


def test_state_rejects_joint_angles():
    """multibody_state is 7 joint angles for an arm; only the pose is valid here."""
    with pytest.raises(ValueError, match="multibody_pose"):
        build_state(np.zeros(7), [0.0], np.zeros(6), [0.0])


def test_action_split_is_the_inverse_of_the_state_layout():
    action = np.arange(ACTION_DIM, dtype=np.float32)
    left_pose, left_gripper, right_pose, right_gripper = split_action(action)

    assert np.allclose(left_pose, action[0:7])
    assert left_gripper == pytest.approx(action[7])
    assert np.allclose(right_pose, action[8:15])
    assert right_gripper == pytest.approx(action[15])
    assert build_state(left_pose, [left_gripper], right_pose, [right_gripper]).tolist() == \
        action.tolist()


def test_action_width_is_checked():
    with pytest.raises(ValueError):
        split_action(np.zeros(14))


def test_cameras_are_matched_by_frame_id_not_by_position():
    frame_ids = ['right_wrist_camera', 'head_camera', 'left_wrist_camera']
    resolved = resolve_camera_indices(frame_ids)
    assert resolved == {'cam_high': 1, 'cam_left_wrist': 2, 'cam_right_wrist': 0}


def test_ambiguous_cameras_are_refused_not_guessed():
    with pytest.raises(ValueError, match='cam_high'):
        resolve_camera_indices(['chain_0', 'chain_1', 'chain_2'])


def test_camera_override_accepts_frame_id_and_index():
    frame_ids = ['chain_0', 'left_wrist', 'right_wrist']
    resolved = resolve_camera_indices(frame_ids, {'cam_high': 'chain_0'})
    assert resolved['cam_high'] == 0
    resolved = resolve_camera_indices(frame_ids, {'cam_high': 0})
    assert resolved['cam_high'] == 0


def test_metadata_check_accepts_the_server_contract():
    check_server_metadata(_metadata())


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"robot_quaternion_format": "wxyz"}, "quaternions"),
        ({"state_dim": 20}, "state_dim"),
        ({"action_dim": 20}, "action_dim"),
        ({"server_normalizes_state": False}, "normalize"),
        (
            {"camera_order": ["video.cam_high", "video.cam_right_wrist", "video.cam_left_wrist"]},
            "camera order",
        ),
    ],
)
def test_metadata_check_rejects_a_mismatched_server(overrides, message):
    with pytest.raises(RuntimeError, match=message):
        check_server_metadata(_metadata(**overrides))


def test_camera_order_matches_the_training_order():
    assert CAMERA_ORDER == ("cam_high", "cam_left_wrist", "cam_right_wrist")
