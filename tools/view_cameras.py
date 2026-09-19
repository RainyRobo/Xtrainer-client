#!/usr/bin/env python3
"""Live view of the three chain cameras the policy uses. Press q to quit.

Run inside the robot Noetic docker (needs DISPLAY)::

    cd /data/arianliu/client
    python3 tools/view_cameras.py

Without a window, write one tiled JPEG and exit::

    python3 tools/view_cameras.py --snapshot /tmp/cams.jpg
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401

import argparse
import time

import cv2
import numpy as np

from vla.xtrainer2_contract import CAMERA_ORDER, resolve_camera_indices


def assigned_views(obs, camera_overrides=None):
    """Map one observation onto the three trained cameras.

    Returns ``(views, error)``. ``views`` is always length 3, each item
    ``(camera, frame_id, index, bgr_or_none)``. ``error`` is set when the
    chain-image list cannot be matched.
    """
    camera_overrides = camera_overrides or {}
    empty = [(camera, None, None, None) for camera in CAMERA_ORDER]
    if obs is None:
        return empty, 'waiting for /robot/observations'
    chain = list(getattr(obs, 'chain_images', None) or [])
    frame_ids = [getattr(image, 'frame_id', None) for image in chain]
    if not chain:
        return empty, 'no chain images in this observation'
    try:
        mapping = resolve_camera_indices(frame_ids, camera_overrides)
    except ValueError as exc:
        return empty, str(exc)
    views = []
    for camera in CAMERA_ORDER:
        index = mapping[camera]
        image = chain[index]
        views.append((
            camera,
            getattr(image, 'frame_id', None),
            index,
            getattr(image, 'color_image', None),
        ))
    return views, None


def extra_chain_views(obs):
    """Every chain image, in publisher order (for ``--all``)."""
    if obs is None:
        return []
    views = []
    for index, image in enumerate(getattr(obs, 'chain_images', None) or []):
        views.append((
            f'chain[{index}]',
            getattr(image, 'frame_id', None),
            index,
            getattr(image, 'color_image', None),
        ))
    return views


def _label(image, text):
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(canvas, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def tile(panels, height):
    """Horizontal strip of labelled BGR frames. Missing frames are grey."""
    strips = []
    for title, frame in panels:
        if frame is None:
            panel = np.full((height, height, 3), 40, dtype=np.uint8)
        else:
            h, w = frame.shape[:2]
            scale = height / float(max(h, 1))
            panel = cv2.resize(
                frame, (max(1, int(w * scale)), height),
                interpolation=cv2.INTER_AREA)
        strips.append(_label(panel, title))
    return np.concatenate(strips, axis=1)


def _title(camera, frame_id, index, hz, frame):
    bits = [camera]
    if index is not None:
        bits.append(f'[{index}]')
    if frame_id:
        bits.append(str(frame_id))
    if hz:
        bits.append(f'{hz:.1f} Hz')
    if frame is not None:
        bits.append(f'mean={float(np.mean(frame)):.0f}')
    else:
        bits.append('no color')
    return '  '.join(bits)


def build_argparser():
    parser = argparse.ArgumentParser(
        description='live view of the three policy cameras (no motion)')
    parser.add_argument('--rate', type=float, default=10.0)
    parser.add_argument('--height', type=int, default=360,
                        help='panel height in pixels')
    parser.add_argument('--all', action='store_true',
                        help='also show every chain image in publisher order')
    parser.add_argument('--snapshot', default='',
                        help='write one tiled JPEG and exit (no window needed)')
    for camera in CAMERA_ORDER:
        parser.add_argument(
            f'--{camera.replace("_", "-")}', default=None,
            help=f'chain-image frame id or index for {camera}')
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    camera_overrides = {
        camera: getattr(args, camera) for camera in CAMERA_ORDER
        if getattr(args, camera) is not None
    }
    want_window = not args.snapshot
    if want_window and not os.environ.get('DISPLAY'):
        print('DISPLAY is empty; OpenCV cannot open a window. '
              'Run this in a desktop session, or pass --snapshot /tmp/cams.jpg.',
              file=sys.stderr)
        return 1

    import rospy
    from data_pipeline.tools.robot_ros_client import RobotRosClient

    rospy.init_node('xtrainer2_cameras', anonymous=True)
    client = RobotRosClient(wait_first_msg=True, frame_id='xtrainer2_cameras')
    rate = rospy.Rate(args.rate)
    last_stamp = {}
    fps = {}
    if want_window:
        print('q in the window, or Ctrl-C, to quit', flush=True)

    while not rospy.is_shutdown():
        obs = client.get_observation(display_image=False)
        views, error = assigned_views(obs, camera_overrides)
        if error:
            rospy.logwarn_throttle(5, '%s', error)

        panels = []
        now = time.time()
        for camera, frame_id, index, frame in views:
            prev = last_stamp.get(camera)
            last_stamp[camera] = now
            if prev:
                fps[camera] = 0.9 * fps.get(camera, 0.0) + 0.1 / max(now - prev, 1e-3)
            panels.append((
                _title(camera, frame_id, index, fps.get(camera, 0.0), frame),
                frame,
            ))
        if args.all:
            for camera, frame_id, index, frame in extra_chain_views(obs):
                panels.append((_title(camera, frame_id, index, 0.0, frame), frame))
        if error:
            panels[0] = (f'{panels[0][0]}  |  {error}', panels[0][1])

        vis = tile(panels, args.height)
        if args.snapshot:
            cv2.imwrite(args.snapshot, vis)
            print(f'wrote {args.snapshot}  {vis.shape[1]}x{vis.shape[0]}', flush=True)
            if error:
                print(error, file=sys.stderr)
                return 1
            return 0

        cv2.imshow('xtrainer cameras', vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        rate.sleep()

    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
