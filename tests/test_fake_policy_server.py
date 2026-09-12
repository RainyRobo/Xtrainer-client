"""The fake server must satisfy the same contract as the real one.

If it drifts, an on-robot rehearsal against it stops proving anything.
"""
import threading

import numpy as np
import pytest

pytest.importorskip('cv2')
pytest.importorskip('websockets.sync.server')

from tools.fake_policy_server import (  # noqa: E402
    FakePolicyServer,
    build_metadata,
    gripper_chunk,
)
from vla.xtrainer2_actions import robot_action_from_16d  # noqa: E402
from vla.xtrainer2_contract import (  # noqa: E402
    ACTION_DIM,
    LEFT_GRIPPER,
    RIGHT_GRIPPER,
    check_server_metadata,
)
from vla.xtrainer2_policy import PolicyBridge  # noqa: E402

from test_policy_bridge_loopback import _Observation  # noqa: E402


def _state():
    return np.array([
        0.1, -0.3, 0.4, 0.0, -1.0, 0.0, 0.0, 1.0,
        0.2, -0.3, 0.4, 0.0, 1.0, 0.0, 0.0, 1.0,
    ], dtype=np.float32)


@pytest.fixture
def server():
    fake = FakePolicyServer('127.0.0.1', 0, chunk=8, delta=0.05,
                            hands=('left',), oscillate=True)
    thread = threading.Thread(target=fake._server.serve_forever, daemon=True)
    thread.start()
    yield fake
    fake._server.shutdown()


def test_metadata_passes_the_client_contract_check():
    check_server_metadata(build_metadata(16))


def test_only_the_selected_gripper_moves_and_poses_are_held():
    state = _state()
    actions, targets = gripper_chunk(state, 8, 0.05, ('left',), -1.0)

    assert actions.shape == (8, ACTION_DIM)
    for step in actions:
        np.testing.assert_allclose(step[0:7], state[0:7])
        np.testing.assert_allclose(step[8:15], state[8:15])
        assert step[RIGHT_GRIPPER] == pytest.approx(state[RIGHT_GRIPPER])
    assert targets['left'] == (pytest.approx(1.0), pytest.approx(0.95))
    assert actions[-1][LEFT_GRIPPER] == pytest.approx(0.95)
    # Ramped, not stepped: every sample sits between start and target.
    assert np.all(np.diff(actions[:, LEFT_GRIPPER]) < 0)


def test_gripper_target_is_clipped_to_the_unit_interval():
    state = _state()
    state[LEFT_GRIPPER] = 0.02
    _, targets = gripper_chunk(state, 4, 0.5, ('left',), -1.0)
    assert targets['left'][1] == pytest.approx(0.0)


def test_a_real_client_can_drive_the_robot_through_this_server(server):
    bridge = PolicyBridge(url=f'ws://127.0.0.1:{server.port}')
    chunk = bridge.infer(bridge.build_example(_Observation()))

    assert chunk.shape == (8, ACTION_DIM)
    # Every returned step must survive the assembly that reaches xtrainer_main.
    for step in chunk:
        built = robot_action_from_16d(step)
        assert [part.name for part in built.actions] == [
            'left_arm', 'left_hand', 'right_arm', 'right_hand']


def test_a_malformed_state_is_reported_not_crashed(server):
    bridge = PolicyBridge(url=f'ws://127.0.0.1:{server.port}')
    example = bridge.build_example(_Observation())
    example['state'] = np.zeros(9, dtype=np.float32)

    with pytest.raises(RuntimeError, match='16-D state'):
        bridge.infer(example)
