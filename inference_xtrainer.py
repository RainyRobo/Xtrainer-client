#!/usr/bin/env python3
"""Drive the arms from a remote StarVLA policy server.

Read ``/robot/observations``, send three camera views plus the 16-D
end-effector state over websocket, then publish the returned 16-D absolute
commands on ``/robot/actions``.

The server owns normalization and wxyz/xyzw conversion. This file only runs
the control loop; message layout lives in ``vla/xtrainer2_actions.py``.
"""
import _bootstrap  # noqa: F401  process-local path only; does not affect other users

import argparse
import collections
import math
import sys
import threading
import time

import rospy
import numpy as np

from data_pipeline.tools.robot_ros_client import RobotRosClient
from vla.xtrainer2_actions import DEFAULT_POSE_FRAME, robot_action_from_16d
from vla.xtrainer2_contract import CAMERA_ORDER, DEFAULT_INSTRUCTION
from vla.xtrainer2_policy import PolicyBridge, check_first_step
from vla.xtrainer2_reset import (
    INIT_POSES, interpolate_to_target, resolve_init_pose, rotation_deg, state_from_observation,
)


class ActionBuffer:
    """Newest inferred chunk; the send loop pops steps from the front."""

    def __init__(self):
        self._lock = threading.Lock()
        self._steps = collections.deque()

    def replace(self, steps):
        with self._lock:
            self._steps = collections.deque(steps)

    def pop(self):
        with self._lock:
            return self._steps.popleft() if self._steps else None


def wait_for_subscriber(client, timeout=5.0):
    deadline = time.time() + timeout
    while client.action_pub.get_num_connections() == 0 and time.time() < deadline:
        rospy.sleep(0.1)
    n = client.action_pub.get_num_connections()
    if n == 0:
        raise RuntimeError('nobody subscribes to /robot/actions; start the robot stack first')
    rospy.loginfo('%d subscriber(s) on /robot/actions', n)


def observation_has_cameras(obs):
    """The first /robot/observations often arrives before any chain image."""
    if obs is None:
        return False
    images = list(getattr(obs, 'chain_images', None) or [])
    if len(images) < len(CAMERA_ORDER):
        return False
    return all(getattr(image, 'color_image', None) is not None for image in images)


def read_observation(client, decoder, start_time):
    """Live observation, or the replayed 20 ms slot for bag playback.

    Returns None until three colour chain images are present, so we never
    handshake-fail on the empty first message.
    """
    if decoder is None:
        obs = client.get_observation(display_image=False)
    else:
        index = round((time.time() - start_time) * 1000 / 20.0)
        if index >= len(decoder.observations):
            return None
        obs = decoder.observations[index]
    if not observation_has_cameras(obs):
        n = len(getattr(obs, 'chain_images', None) or []) if obs is not None else 0
        rospy.logwarn_throttle(
            5, 'waiting for %d chain images (have %d)', len(CAMERA_ORDER), n)
        return None
    return obs


def publish(client, vector, pose_frame):
    client.send_action(robot_action_from_16d(
        vector, pose_frame=pose_frame, timestamp=rospy.Time.now().to_sec()))


def move_to_init_target(client, send_hz, pose_frame, target, name, duration=2.0):
    """Stream an interpolated trajectory to ``target`` before inference."""
    obs = client.get_observation(display_image=False)
    if obs is None:
        raise RuntimeError('no /robot/observations; cannot reset to the start pose')
    current = state_from_observation(obs)
    left_deg = rotation_deg(current[3:7], target[3:7])
    right_deg = rotation_deg(current[11:15], target[11:15])
    left_cm = 100.0 * float(np.linalg.norm(current[0:3] - target[0:3]))
    right_cm = 100.0 * float(np.linalg.norm(current[8:11] - target[8:11]))
    n_steps = max(2, int(round(send_hz * duration)))
    traj = interpolate_to_target(current, target, n_steps)
    rospy.loginfo(
        'resetting to init pose %r in frame %r over %.1fs: '
        'left %.1f cm / %.1f deg, right %.1f cm / %.1f deg',
        name, pose_frame, duration, left_cm, left_deg, right_cm, right_deg)
    rate = rospy.Rate(send_hz)
    for step in traj:
        if rospy.is_shutdown() or client.action_pub.get_num_connections() == 0:
            raise RuntimeError('lost /robot/actions subscriber during reset')
        publish(client, step, pose_frame)
        rate.sleep()
    rospy.sleep(1.0)


def hold_and_send(client, vector, pose_frame, send_hz, policy_hz):
    """Repeat one 10 Hz policy step at the robot's command rate (100 Hz)."""
    repeats = max(1, int(round(send_hz / policy_hz)))
    rate = rospy.Rate(send_hz)
    for _ in range(repeats):
        if rospy.is_shutdown() or client.action_pub.get_num_connections() == 0:
            return False
        publish(client, vector, pose_frame)
        rate.sleep()
    return True


def run_serial(bridge, client, decoder, policy_hz, send_hz, pose_frame,
               steps_per_chunk, max_jump):
    start_time = time.time()
    while not rospy.is_shutdown():
        obs = read_observation(client, decoder, start_time)
        if obs is None:
            rospy.sleep(1.0 / policy_hz)
            continue

        example = bridge.build_example(obs)
        chunk = bridge.infer(example)
        check_first_step(chunk[0], example['state'], max_jump)

        n_steps = min(steps_per_chunk or len(chunk), len(chunk))
        for step in chunk[:n_steps]:
            if not hold_and_send(client, step, pose_frame, send_hz, policy_hz):
                rospy.logerr('lost /robot/actions subscriber; stopping')
                return
    rospy.loginfo('serial policy loop finished')


def infer_loop(bridge, client, decoder, buffer, infer_hz, policy_hz, max_jump):
    start_time = time.time()
    while not rospy.is_shutdown():
        loop_start = time.time()
        obs = read_observation(client, decoder, start_time)
        if obs is None:
            time.sleep(1.0 / infer_hz)
            continue

        example = bridge.build_example(obs)
        t_obs = time.time()
        chunk = bridge.infer(example)
        check_first_step(chunk[0], example['state'], max_jump)

        skipped = math.ceil((time.time() - t_obs) * policy_hz)
        if skipped >= len(chunk):
            rospy.logwarn('chunk of %d steps shorter than %d elapsed steps',
                          len(chunk), skipped)
        else:
            buffer.replace(chunk[skipped:])

        leftover = (1.0 / infer_hz) - (time.time() - loop_start)
        if leftover > 0:
            time.sleep(leftover)


def send_loop(client, buffer, pose_frame, policy_hz, send_hz):
    """Advance through the buffer at policy_hz, publishing each step at send_hz."""
    repeats = max(1, int(round(send_hz / policy_hz)))
    rate = rospy.Rate(send_hz)
    current = None
    remaining = 0
    while not rospy.is_shutdown():
        if client.action_pub.get_num_connections() == 0:
            rospy.logerr('lost /robot/actions subscriber; stopping')
            return
        if remaining <= 0:
            nxt = buffer.pop()
            if nxt is not None:
                current = nxt
                remaining = repeats
        if current is not None:
            publish(client, current, pose_frame)
            remaining -= 1
        rate.sleep()


def build_argparser():
    parser = argparse.ArgumentParser(description='xtrainer2 on-robot VLA client')
    parser.add_argument('--url', required=True,
                        help='policy server, e.g. ws://10.0.0.5:10093')
    parser.add_argument('--instruction', default=DEFAULT_INSTRUCTION,
                        help='English instruction the model was trained on')
    parser.add_argument('--unnorm_key', default=None,
                        help='only needed for multi-dataset checkpoints')
    parser.add_argument('--rate', type=float, default=10.0,
                        help='how fast to advance through the action chunk (dataset Hz)')
    parser.add_argument('--send-rate', type=float, default=100.0,
                        help='publish rate on /robot/actions; xtrainer_main expects ~100 Hz')
    parser.add_argument('--infer-rate', type=float, default=4.0,
                        help='inference rate in --parallel mode')
    parser.add_argument('--steps-per-chunk', type=int, default=0,
                        help='serial mode: steps executed per chunk (0 = whole chunk)')
    parser.add_argument('--parallel', action='store_true',
                        help='infer and execute in separate threads')
    parser.add_argument('--pose-frame', default=DEFAULT_POSE_FRAME,
                        choices=[DEFAULT_POSE_FRAME, ''],
                        help="pose frame; TorsoEE is what /robot/observations reports")
    parser.add_argument('--max-first-step-jump', type=float, default=0.15,
                        help='metres; refuse a chunk whose first step is farther '
                             'than this from the current pose (0 disables)')
    parser.add_argument('--bag-path', default='',
                        help='replay observations from a rosbag/mcap instead of the live robot')
    parser.add_argument('--init-pose', default='default', choices=sorted(INIT_POSES),
                        help='named start pose to reset to before inference')
    parser.add_argument('--skip-reset', action='store_true',
                        help='do not move to the start pose before the first policy command')
    for camera in CAMERA_ORDER:
        parser.add_argument(f'--{camera.replace("_", "-")}', default=None,
                            help=f'chain-image frame id or index for {camera}')
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    rospy.init_node('xtrainer2_inference', anonymous=True)

    camera_overrides = {
        camera: getattr(args, camera) for camera in CAMERA_ORDER
        if getattr(args, camera) is not None
    }
    bridge = PolicyBridge(
        url=args.url,
        instruction=args.instruction,
        unnorm_key=args.unnorm_key,
        camera_overrides=camera_overrides,
    )
    rospy.loginfo('pose frame %r, policy %.1f Hz, send %.1f Hz',
                  args.pose_frame, args.rate, args.send_rate)

    decoder = None
    if args.bag_path:
        from data_pipeline.tools.robot_rosbag_decoder import RobotRosbagDecoder
        decoder = RobotRosbagDecoder()
        decoder.decode(args.bag_path, insert_previous_data=True)
        rospy.loginfo('replaying observations from %s', args.bag_path)

    client = RobotRosClient(frame_id='xtrainer2_inference')
    wait_for_subscriber(client)
    if not args.skip_reset:
        move_to_init_target(
            client, args.send_rate, args.pose_frame,
            resolve_init_pose(args.init_pose), args.init_pose)

    if args.parallel:
        buffer = ActionBuffer()
        sender = threading.Thread(
            target=send_loop,
            args=(client, buffer, args.pose_frame, args.rate, args.send_rate),
            daemon=True)
        inferer = threading.Thread(
            target=infer_loop,
            args=(bridge, client, decoder, buffer, args.infer_rate, args.rate,
                  args.max_first_step_jump),
            daemon=True)
        sender.start()
        inferer.start()
        rospy.spin()
        return 0

    run_serial(bridge, client, decoder, args.rate, args.send_rate,
               args.pose_frame, args.steps_per_chunk, args.max_first_step_jump)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
