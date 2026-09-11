"""End-to-end check of the request/response path, without ROS or a GPU.

A stub server speaks the real wire format (msgpack, the server's response
envelope), so this catches an envelope or key rename on either side. What it
cannot catch is a wrong checkpoint; that is what the metadata check is for.
"""

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

CLIENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLIENT_ROOT))
sys.path.insert(0, str(CLIENT_ROOT / "vendor"))

cv2 = pytest.importorskip("cv2")
websockets_sync_server = pytest.importorskip("websockets.sync.server")

from vla.third_party.openpi_client import msgpack_numpy  # noqa: E402
from vla.xtrainer2_contract import ACTION_DIM, CAMERA_ORDER, DEFAULT_INSTRUCTION  # noqa: E402
from vla.xtrainer2_policy import PolicyBridge, check_first_step  # noqa: E402

CHUNK = 16
METADATA = {
    "action_chunk_size": CHUNK,
    "camera_order": ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"],
    "state_dim": 16,
    "action_dim": ACTION_DIM,
    "robot_quaternion_format": "xyzw",
    "model_quaternion_format": "wxyz",
    "server_normalizes_state": True,
    "image_format": "HWC_RGB_uint8",
}


class _ChainImage:
    def __init__(self, frame_id, color_image):
        self.frame_id = frame_id
        self.color_image = color_image


class _Component:
    def __init__(self, multibody_pose=None, multibody_state=None):
        self.multibody_pose = multibody_pose
        self.multibody_state = multibody_state


class _Observation:
    """The parts of ``RobotObservation`` the bridge touches."""

    def __init__(self):
        # Deliberately not in training order: the bridge must reorder them.
        self.chain_images = [
            _ChainImage('right_wrist_camera', np.full((480, 848, 3), 30, dtype=np.uint8)),
            _ChainImage('head_camera', np.full((480, 848, 3), 10, dtype=np.uint8)),
            _ChainImage('left_wrist_camera', np.full((480, 848, 3), 20, dtype=np.uint8)),
        ]
        self._components = {
            'left_arm': _Component(multibody_pose=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])),
            'right_arm': _Component(multibody_pose=np.array([0.4, 0.5, 0.6, 0.0, 0.0, 1.0, 0.0])),
            'left_hand': _Component(multibody_state=np.array([0.7])),
            'right_hand': _Component(multibody_state=np.array([0.8])),
        }

    def get_observation(self, name):
        return self._components[name]


class _StubServer:
    """Replays the policy server's envelope and records what it received."""

    def __init__(self):
        self.requests = []
        self._packer = msgpack_numpy.Packer()
        self._server = websockets_sync_server.serve(
            self._handler, '127.0.0.1', 0, compression=None, max_size=None
        )
        self.port = self._server.socket.getsockname()[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def _handler(self, connection):
        connection.send(self._packer.pack(METADATA))
        for message in connection:
            request = msgpack_numpy.unpackb(message)
            self.requests.append(request)
            state = np.asarray(request["examples"][0]["state"], dtype=np.float32)
            actions = np.tile(state, (CHUNK, 1))
            connection.send(self._packer.pack({
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": request.get("request_id", "default"),
                "data": {"actions": actions[None, ...]},
            }))

    def close(self):
        self._server.shutdown()


@pytest.fixture
def server():
    stub = _StubServer()
    yield stub
    stub.close()


def test_round_trip_sends_the_trained_layout_and_parses_the_envelope(server):
    bridge = PolicyBridge(url=f'ws://127.0.0.1:{server.port}')
    example = bridge.build_example(_Observation())
    actions = bridge.infer(example)

    assert actions.shape == (CHUNK, ACTION_DIM)
    request = server.requests[0]["examples"][0]

    assert request["lang"] == DEFAULT_INSTRUCTION
    assert len(request["image"]) == len(CAMERA_ORDER)
    for image in request["image"]:
        assert image.shape == (224, 224, 3)
        assert image.dtype == np.uint8

    # Views are ordered cam_high, cam_left_wrist, cam_right_wrist regardless of
    # the order the robot published them; the fixture fills each with a
    # distinct grey level so the reordering is observable.
    assert [int(image[0, 0, 0]) for image in request["image"]] == [10, 20, 30]

    assert np.allclose(request["state"][0:3], [0.1, 0.2, 0.3])
    assert request["state"][7] == pytest.approx(0.7)
    assert np.allclose(request["state"][8:11], [0.4, 0.5, 0.6])
    assert request["state"][15] == pytest.approx(0.8)


def test_bgr_frames_are_sent_as_rgb(server):
    """cv_bridge hands over BGR; training frames were decoded as RGB."""
    bridge = PolicyBridge(url=f'ws://127.0.0.1:{server.port}')
    observation = _Observation()
    for chain_image in observation.chain_images:
        chain_image.color_image = np.zeros((480, 848, 3), dtype=np.uint8)
        chain_image.color_image[..., 0] = 200  # blue in OpenCV order

    bridge.infer(bridge.build_example(observation))
    image = server.requests[0]["examples"][0]["image"][0]
    assert image[0, 0].tolist() == [0, 0, 200]


def test_server_error_is_raised_not_executed(server):
    bridge = PolicyBridge(url=f'ws://127.0.0.1:{server.port}')

    class _Failing:
        def get_server_metadata(self):
            return METADATA

        def infer(self, request):
            return {"status": "error", "ok": False, "type": "inference_result",
                    "error": {"message": "unnorm_key not specified"}}

    bridge.client = _Failing()
    with pytest.raises(RuntimeError, match="unnorm_key"):
        bridge.infer(bridge.build_example(_Observation()))


def test_a_teleporting_first_step_is_refused():
    state = np.zeros(16, dtype=np.float32)
    action = np.zeros(16, dtype=np.float32)
    action[0] = 0.9  # 90 cm away from the current left pose

    with pytest.raises(RuntimeError, match="left arm"):
        check_first_step(action, state, max_jump=0.15)
    check_first_step(action, state, max_jump=0.0)  # disabled
