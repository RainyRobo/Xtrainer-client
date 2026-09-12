"""Guards for the pre-flight probe, without ROS or a GPU."""

import numpy as np

from tools.probe_xtrainer import (
    MockObservation,
    observation_state,
    summarize_actions,
    summarize_observation,
)
from vla.xtrainer2_contract import CAMERA_ORDER, STATE_DIM


def test_mock_observation_builds_a_valid_16d_state():
    obs = MockObservation()
    dump = summarize_observation(obs)

    assert dump['mapping_error'] is None
    assert dump['mapping'] == {
        'cam_high': 1,
        'cam_left_wrist': 2,
        'cam_right_wrist': 0,
    }
    state = dump['state']
    assert state.shape == (STATE_DIM,)
    assert np.allclose(state, observation_state(obs))
    assert 'left_arm' in '\n'.join(dump['lines'])
    for camera in CAMERA_ORDER:
        assert camera in dump['mapping']


def test_summarize_actions_flags_a_teleport():
    state = np.zeros(16, dtype=np.float32)
    actions = np.zeros((4, 16), dtype=np.float32)
    actions[:, 0] = 0.4
    lines = summarize_actions(actions, state, max_jump=0.15)
    joined = '\n'.join(lines)
    assert 'EXCEEDS' in joined
    assert 'left: first xyz jump 0.4000 m' in joined
