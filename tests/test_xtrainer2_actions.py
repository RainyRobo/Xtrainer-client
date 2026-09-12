"""The ROS action layout that crashed xtrainer_main when it was wrong."""
import numpy as np
import pytest

from vla.xtrainer2_actions import (
    BODY_ORDER,
    DEFAULT_POSE_FRAME,
    robot_action_from_16d,
)


def _action(updates=None):
    vector = np.array([
        0.1, -0.3, 0.4, 0.0, -1.0, 0.0, 0.0, 0.2,
        0.2, -0.3, 0.4, 0.0, 1.0, 0.0, 0.0, 0.9,
    ], dtype=np.float32)
    if updates:
        for index, value in updates.items():
            vector[index] = value
    return vector


def test_components_match_observation_order():
    built = robot_action_from_16d(_action())
    assert tuple(c.name for c in built.actions) == BODY_ORDER


def test_hands_reuse_the_arm_pose_and_never_leave_a_zero_quaternion():
    built = robot_action_from_16d(_action(), pose_frame=DEFAULT_POSE_FRAME)
    left_arm, left_hand, right_arm, right_hand = built.actions

    np.testing.assert_allclose(left_hand.pose_command.pose, left_arm.pose_command.pose)
    np.testing.assert_allclose(right_hand.pose_command.pose, right_arm.pose_command.pose)
    assert left_hand.pose_command.frame == DEFAULT_POSE_FRAME
    assert np.linalg.norm(left_hand.pose_command.pose[3:]) > 0.5
    np.testing.assert_allclose(left_hand.joint_commands, [0.2])
    np.testing.assert_allclose(right_hand.joint_commands, [0.9])
    assert left_arm.joint_commands is None


def test_grippers_are_clipped_to_unit_interval():
    built = robot_action_from_16d(_action({7: -0.4, 15: 1.7}))
    assert built.actions[1].joint_commands[0] == 0.0
    assert built.actions[3].joint_commands[0] == 1.0


def test_zero_quaternion_is_refused():
    with pytest.raises(ValueError, match='quaternion'):
        robot_action_from_16d(_action({3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}))
