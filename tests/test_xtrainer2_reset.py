"""Start-pose interpolation, without ROS."""
import numpy as np
import pytest

from vla.xtrainer2_actions import robot_action_from_16d
from vla.xtrainer2_contract import ACTION_DIM, build_state
from vla.xtrainer2_reset import (
    INIT_POSES,
    INIT_TARGET,
    interpolate_to_target,
    resolve_init_pose,
    rotation_deg,
    state_from_observation,
)


class _Component:
    def __init__(self, pose=None, state=None):
        self.multibody_pose = pose
        self.multibody_state = state


class _Observation:
    def __init__(self):
        self._components = {
            'left_arm': _Component(pose=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])),
            'right_arm': _Component(pose=np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])),
            'left_hand': _Component(state=np.array([0.2])),
            'right_hand': _Component(state=np.array([0.3])),
        }

    def get_observation(self, name):
        return self._components[name]


def test_init_target_is_a_valid_16d_action():
    assert INIT_TARGET.shape == (ACTION_DIM,)
    built = robot_action_from_16d(INIT_TARGET)
    assert len(built.actions) == 4
    np.testing.assert_allclose(built.actions[1].joint_commands, [INIT_TARGET[7]])


def test_named_poses_are_switchable_and_unit_quaternions():
    assert set(INIT_POSES) >= {'default', 'eyebrow', 'pack_phone'}
    for name in ('eyebrow', 'pack_phone'):
        pose = resolve_init_pose(name)
        assert pose.shape == (ACTION_DIM,)
        assert np.linalg.norm(pose[3:7]) == pytest.approx(1.0, abs=1e-3)
        assert np.linalg.norm(pose[11:15]) == pytest.approx(1.0, abs=1e-3)
        robot_action_from_16d(pose)
        # Episode dumps are wxyz. After conversion the wrist matches the
        # TorsoEE start pose; the unconverted dump is about 90 degrees off.
        assert rotation_deg(pose[3:7], INIT_TARGET[3:7]) < 15.0
        assert rotation_deg(pose[11:15], INIT_TARGET[11:15]) < 15.0
    with pytest.raises(KeyError, match='unknown init pose'):
        resolve_init_pose('missing')


def test_interpolate_reaches_the_target_and_keeps_unit_quaternions():
    current = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ], dtype=np.float64)
    traj = interpolate_to_target(current, INIT_TARGET, n_steps=20)
    assert traj.shape == (20, ACTION_DIM)
    np.testing.assert_allclose(traj[-1, :3], INIT_TARGET[:3], atol=1e-9)
    np.testing.assert_allclose(traj[-1, 8:11], INIT_TARGET[8:11], atol=1e-9)
    np.testing.assert_allclose(traj[-1, 7], INIT_TARGET[7])
    np.testing.assert_allclose(traj[-1, 15], INIT_TARGET[15])
    for step in traj:
        assert np.linalg.norm(step[3:7]) == pytest.approx(1.0, abs=1e-6)
        assert np.linalg.norm(step[11:15]) == pytest.approx(1.0, abs=1e-6)


def test_slerp_takes_the_short_arc_when_quaternion_signs_flip():
    current = np.zeros(ACTION_DIM, dtype=np.float64)
    current[6] = 1.0
    current[14] = 1.0
    target = current.copy()
    target[3:7] = -current[3:7]
    traj = interpolate_to_target(current, target, n_steps=4)
    # Same orientation after a sign flip: every step stays near identity.
    for step in traj:
        assert abs(step[6]) == pytest.approx(1.0, abs=1e-5)


def test_state_from_observation_uses_ee_pose_not_joints():
    state = state_from_observation(_Observation())
    expected = build_state(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.2],
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.3],
    )
    np.testing.assert_allclose(state, expected)
