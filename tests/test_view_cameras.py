"""Camera viewer layout, without ROS or a window."""

from tools.probe_xtrainer import MockObservation
from tools.view_cameras import assigned_views, extra_chain_views, tile
from vla.xtrainer2_contract import CAMERA_ORDER


def test_assigned_views_follow_training_order():
    views, error = assigned_views(MockObservation())
    assert error is None
    assert [camera for camera, *_ in views] == list(CAMERA_ORDER)
    assert [frame_id for _, frame_id, *_ in views] == [
        'head_camera', 'left_wrist_camera', 'right_wrist_camera',
    ]
    assert [index for _, _, index, _ in views] == [1, 2, 0]
    assert all(frame is not None for *_, frame in views)


def test_assigned_views_placeholder_when_empty():
    views, error = assigned_views(None)
    assert 'waiting' in error
    assert len(views) == 3
    assert all(frame is None for *_, frame in views)


def test_tile_keeps_three_panels():
    obs = MockObservation()
    views, _ = assigned_views(obs)
    vis = tile([(camera, frame) for camera, *_, frame in views], height=120)
    assert vis.shape[0] == 120
    assert vis.shape[2] == 3
    assert vis.shape[1] > 120 * 2


def test_extra_chain_views_are_publisher_order():
    extras = extra_chain_views(MockObservation())
    assert [frame_id for _, frame_id, *_ in extras] == [
        'right_wrist_camera', 'head_camera', 'left_wrist_camera',
    ]
