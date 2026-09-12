#!/usr/bin/env python3
"""Live view of /robot/observations cameras. Press q to quit.

Run inside the robot Noetic docker (needs DISPLAY)::

    cd /data/arianliu/client
    python3 tools/view_cameras.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401

import argparse
import time

import cv2
import numpy as np


def _label(image, text):
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(canvas, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _tile(images, height):
    panels = []
    for title, frame in images:
        if frame is None:
            panel = np.zeros((height, height, 3), dtype=np.uint8)
            panel[:] = (40, 40, 40)
        else:
            h, w = frame.shape[:2]
            scale = height / float(h)
            panel = cv2.resize(frame, (int(w * scale), height), interpolation=cv2.INTER_AREA)
        panels.append(_label(panel, title))
    return np.concatenate(panels, axis=1)


def main():
    parser = argparse.ArgumentParser(description='live robot camera view (no motion)')
    parser.add_argument('--rate', type=float, default=10.0)
    parser.add_argument('--height', type=int, default=360,
                        help='panel height in pixels')
    args = parser.parse_args()

    if not os.environ.get('DISPLAY'):
        print('DISPLAY is empty; OpenCV cannot open a window. '
              'Run this in a desktop session (or export DISPLAY=:0).', file=sys.stderr)
        return 1

    import rospy
    from data_pipeline.tools.robot_ros_client import RobotRosClient

    rospy.init_node('xtrainer2_cameras', anonymous=True)
    client = RobotRosClient(wait_first_msg=True, frame_id='xtrainer2_cameras')
    rate = rospy.Rate(args.rate)
    last_stamp = {}
    fps = {}
    print('q in the window, or Ctrl-C, to quit', flush=True)

    while not rospy.is_shutdown():
        obs = client.get_observation(display_image=False)
        panels = []
        if obs is None:
            panels.append(('waiting /robot/observations', None))
        else:
            sources = list(obs.chain_images) + list(obs.camera_images)
            if not sources:
                panels.append(('no images in this observation', None))
            for img in sources:
                frame = getattr(img, 'color_image', None)
                if frame is None:
                    frame = getattr(img, 'image', None)
                name = getattr(img, 'frame_id', '?') or '?'
                now = time.time()
                prev = last_stamp.get(name)
                last_stamp[name] = now
                if prev:
                    fps[name] = 0.9 * fps.get(name, 0.0) + 0.1 / max(now - prev, 1e-3)
                hz = fps.get(name, 0.0)
                mean = float(np.mean(frame)) if frame is not None else 0.0
                title = f'{name}  {hz:.1f} Hz  mean={mean:.0f}'
                panels.append((title, frame))

        vis = _tile(panels, args.height)
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
