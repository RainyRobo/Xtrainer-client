#!/usr/bin/env python3
"""Drive the arms from a remote StarVLA policy server.

Loop: read a ROS observation, send three camera views plus the 16-D
end-effector state over websocket, receive a chunk of absolute 16-D
end-effector commands, publish them at the control rate.

The server owns normalization and the wxyz/xyzw quaternion conversion, so
everything on this side stays in the robot's own units. The layout is defined
in ``vla/xtrainer2_contract.py``.
"""
import _bootstrap  # noqa: F401  process-local path only; does not affect other users

import argparse
import math
import threading
import time

import numpy as np
import rospy

from data_pipeline.tools.robot_action import ComponentAction, RobotAction
from data_pipeline.tools.robot_ros_client import RobotRosClient
from vla.xtrainer2_contract import CAMERA_ORDER, DEFAULT_INSTRUCTION, split_action
from vla.xtrainer2_policy import PolicyBridge, check_first_step

_G_ACTION_LIST = []
lock = threading.Lock()


def generate_action(cur_action, pose_frame):
    """Turn one 16-D server action into a ROS ``RobotAction``."""
    left_pose, left_gripper, right_pose, right_gripper = split_action(cur_action)

    actions = RobotAction()
    for name, pose in (('left_arm', left_pose), ('right_arm', right_pose)):
        arm_action = ComponentAction()
        arm_action.name = name
        arm_action.pose_command = ComponentAction.Pose(np.array(pose), pose_frame)
        arm_action.joint_commands = None
        arm_action.duration = 0.0
        actions.actions.append(arm_action)

    # Grippers stay continuous; binarizing them changes the learned behaviour.
    for name, opening in (('left_hand', left_gripper), ('right_hand', right_gripper)):
        hand_action = ComponentAction()
        hand_action.name = name
        hand_action.pose_command = None
        hand_action.joint_commands = np.array([opening])
        hand_action.duration = 0.0
        actions.actions.append(hand_action)

    actions.timestamp = rospy.Time.now().to_sec()
    return actions


def read_observation(client, decoder, start_time):
    """Live observation, or the replayed one for the elapsed 20 ms slot."""
    if decoder is None:
        return client.get_observation(display_image=False)
    index = round((time.time() - start_time) * 1000 / 20.0)
    if index >= len(decoder.observations):
        return None
    return decoder.observations[index]


def deploy_policy(bridge, client, decoder, rate_hz, pose_frame, steps_per_chunk, max_jump):
    """Serial loop: one inference, then execute part of the chunk, then repeat."""
    rate = rospy.Rate(rate_hz)
    start_time = time.time()

    while not rospy.is_shutdown():
        obs = read_observation(client, decoder, start_time)
        if obs is None:
            rate.sleep()
            continue

        example = bridge.build_example(obs)
        action_chunk = bridge.infer(example)
        check_first_step(action_chunk[0], example["state"], max_jump)

        executed = min(steps_per_chunk or bridge.action_chunk_size, len(action_chunk))
        for act_i in range(executed):
            client.send_action(generate_action(action_chunk[act_i], pose_frame))
            rate.sleep()
    print('deploy policy done.')


def policy_infer_thread(bridge, client, decoder, infer_rate, act_rate, max_jump):
    global _G_ACTION_LIST
    start_time = time.time()
    while not rospy.is_shutdown():
        loop_start = time.time()
        obs = read_observation(client, decoder, start_time)
        if obs is None:
            time.sleep(1.0 / infer_rate)
            continue

        example = bridge.build_example(obs)
        obs_time = time.time()
        action_chunk = bridge.infer(example)
        check_first_step(action_chunk[0], example["state"], max_jump)
        action_chunk = action_chunk.tolist()

        with lock:
            if len(_G_ACTION_LIST) == 0:
                _G_ACTION_LIST = action_chunk.copy()
            else:
                # Drop the steps that elapsed while the server was thinking.
                removed_len = math.ceil((time.time() - obs_time) * act_rate)
                if removed_len < len(action_chunk):
                    _G_ACTION_LIST = action_chunk[removed_len:]
                else:
                    print(f'Predicted action length {len(action_chunk)} too short '
                          f'for {removed_len} elapsed steps')

        sleep_sec = (1 / infer_rate) - (time.time() - loop_start)
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    print('deploy policy done.')


def send_action_thread(client, rate_hz, pose_frame):
    global _G_ACTION_LIST
    ros_rate = rospy.Rate(rate_hz)
    while not rospy.is_shutdown():
        ros_rate.sleep()
        with lock:
            cur_action = _G_ACTION_LIST.pop(0) if _G_ACTION_LIST else None
        if cur_action is None:
            print('Empty action list')
            continue
        client.send_action(generate_action(cur_action, pose_frame))
    print('send action done.')


def build_argparser():
    parser = argparse.ArgumentParser(description='xtrainer2 on-robot VLA client')
    parser.add_argument('--url', type=str, required=True,
                        help='policy server, e.g. ws://10.0.0.5:10093')
    parser.add_argument('--instruction', type=str, default=DEFAULT_INSTRUCTION,
                        help='must be the English instruction the model was trained on')
    parser.add_argument('--unnorm_key', type=str, default=None,
                        help='only needed for multi-dataset checkpoints')
    parser.add_argument('--rate', type=float, default=10,
                        help='action execution rate; the datasets are 10 Hz')
    parser.add_argument('--infer-rate', type=float, default=4.0,
                        help='inference rate in --parallel mode')
    parser.add_argument('--steps-per-chunk', type=int, default=0,
                        help='steps executed per chunk in serial mode (0 = whole chunk)')
    parser.add_argument('--parallel', action='store_true',
                        help='infer and execute in separate threads')
    parser.add_argument('--pose-frame', type=str, default='TorsoEe',
                        choices=['TorsoEe', 'TorsoTool'],
                        help='frame of the commanded end-effector pose')
    parser.add_argument('--max-first-step-jump', type=float, default=0.15,
                        help='metres; refuse a chunk whose first step is farther '
                             'than this from the current pose (0 disables)')
    parser.add_argument('--bag-path', type=str, default='',
                        help='replay observations from a rosbag/mcap instead of the live robot')
    for camera in CAMERA_ORDER:
        parser.add_argument(f'--{camera.replace("_", "-")}', type=str, default=None,
                            help=f'chain-image frame id or index for {camera}')
    return parser


def main():
    args = build_argparser().parse_args()
    rospy.init_node('robot_ros_client_example', anonymous=True)
    rospy.loginfo('Ros node has been initialized')

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
    print(f'[client] pose frame {args.pose_frame}, execution rate {args.rate} Hz')

    decoder = None
    if args.bag_path:
        from data_pipeline.tools.robot_rosbag_decoder import RobotRosbagDecoder
        decoder = RobotRosbagDecoder()
        decoder.decode(args.bag_path, insert_previous_data=True)
        print(f'[client] replaying observations from {args.bag_path}')

    client = RobotRosClient(frame_id='robot_ros_client')
    if args.parallel:
        action_thread = threading.Thread(
            target=send_action_thread, args=(client, args.rate, args.pose_frame))
        action_thread.daemon = True
        action_thread.start()

        policy_thread = threading.Thread(
            target=policy_infer_thread,
            args=(bridge, client, decoder, args.infer_rate, args.rate,
                  args.max_first_step_jump))
        policy_thread.daemon = True
        policy_thread.start()

        rospy.spin()
        action_thread.join()
        policy_thread.join()
    else:
        deploy_policy(bridge, client, decoder, args.rate, args.pose_frame,
                      args.steps_per_chunk, args.max_first_step_jump)


if __name__ == '__main__':
    main()
