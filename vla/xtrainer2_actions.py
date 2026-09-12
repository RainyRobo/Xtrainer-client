"""Turn a 16-D policy action into the ``RobotAction`` xtrainer_main accepts.

``xtrainer_main`` indexes ``component_actions[i]`` in the same order as
``/robot/observations`` and runs ``QuaternionToRotationMatrix`` on every
``pose_command``. A missing pose serialises as (0, 0, 0, 0) and aborts the
process; a short list segfaults. Hands therefore carry the matching arm pose.
"""
import numpy as np

from data_pipeline.tools.robot_action import ComponentAction, RobotAction
from vla.xtrainer2_contract import split_action

#: Same order the robot publishes in ``/robot/observations``.
BODY_ORDER = ('left_arm', 'left_hand', 'right_arm', 'right_hand')

#: Only frame id present in the xtrainer_main binary.
DEFAULT_POSE_FRAME = 'TorsoEE'


def robot_action_from_16d(action, pose_frame=DEFAULT_POSE_FRAME, duration=0.0,
                          timestamp=None):
    """Build one four-component command from a 16-D xyzw action."""
    left_pose, left_gripper, right_pose, right_gripper = split_action(action)
    poses = {
        'left': np.asarray(left_pose, dtype=np.float64).reshape(7),
        'right': np.asarray(right_pose, dtype=np.float64).reshape(7),
    }
    grippers = {
        'left': float(np.clip(left_gripper, 0.0, 1.0)),
        'right': float(np.clip(right_gripper, 0.0, 1.0)),
    }
    for side, pose in poses.items():
        if not np.any(pose[3:]):
            raise ValueError(f'{side} arm quaternion is all zeros')

    actions = RobotAction()
    for side in ('left', 'right'):
        arm = ComponentAction()
        arm.name = f'{side}_arm'
        arm.pose_command = ComponentAction.Pose(poses[side], pose_frame)
        arm.joint_commands = None
        arm.duration = duration
        actions.actions.append(arm)

        hand = ComponentAction()
        hand.name = f'{side}_hand'
        hand.pose_command = ComponentAction.Pose(poses[side], pose_frame)
        hand.joint_commands = np.array([grippers[side]], dtype=np.float64)
        hand.duration = duration
        actions.actions.append(hand)

    names = tuple(component.name for component in actions.actions)
    if names != BODY_ORDER:
        raise RuntimeError(f'component order {names} != {BODY_ORDER}')

    actions.timestamp = timestamp
    return actions
