#!/usr/bin/env python3
import _bootstrap  # noqa: F401  process-local path only; does not affect other users

import time
import math
import threading
import argparse
import numpy as np
import rospy
import cv2

from data_pipeline.tools.robot_action import RobotAction, ComponentAction
from data_pipeline.tools.robot_ros_client import RobotRosClient

from vla.third_party.openpi_client import websocket_client_policy as _websocket_client_policy

url : str = "server_url"
policy_client = _websocket_client_policy.WebsocketClientPolicy(url)
_G_ACTION_LIST = []

START_TIME = None


def infer(obs, policy_client, use_tactile):
    for cam_img in obs.camera_images:
        if cam_img.image is None:
            continue

    for chain_img in obs.chain_images:
        if chain_img.color_image is None:
            continue

    cam_left_wrist = np.transpose(cv2.resize(obs.chain_images[1].color_image, (224, 224)), (2, 0, 1))
    cam_right_wrist = np.transpose(cv2.resize(obs.chain_images[2].color_image, (224, 224)), (2, 0, 1))
    resized_color = cv2.resize(obs.chain_images[0].color_image, (224, 224))
    cam_high = np.transpose(resized_color, (2, 0, 1))

    obs_ee_pose_left = np.array(obs.get_observation(
        name='left_arm').multibody_state)  # (7,)
    obs_gripper_left = obs.get_observation(
        name='left_hand').multibody_state  # (1,)
    obs_ee_pose_right = np.array(obs.get_observation(
        name='right_arm').multibody_state)  # (7,)
    obs_gripper_right = obs.get_observation(
        name='right_hand').multibody_state  # (1,)

    state = np.concatenate((obs_ee_pose_left, obs_gripper_left, obs_ee_pose_right, obs_gripper_right), axis=0)

    element = {
        "state": state,
        "images": {
            "cam_high": cam_high,
            "cam_left_wrist": cam_left_wrist,
            "cam_right_wrist": cam_right_wrist,
        },
        "prompt": "put the objects in the box",
    }

    start = time.time()
    action_chunk = policy_client.infer(element)["actions"]
    end = time.time()

    print('action_chunk', action_chunk.shape, 'time', end - start)  # (50, 16)
    return action_chunk


def generate_action(cur_action):
    actions = RobotAction()
    act_ee_pose_left = cur_action[0:6]
    act_gripper_left = cur_action[6]
    act_ee_pose_right = cur_action[7:13]
    act_gripper_right = cur_action[13]

    left_arm_action = ComponentAction()
    left_arm_action.name = 'left_arm'
    left_arm_action.joint_commands = np.array(act_ee_pose_left)
    left_arm_action.pose_command = None
    left_arm_action.duration = 0.0
    actions.actions.append(left_arm_action)

    right_arm_action = ComponentAction()
    right_arm_action.name = 'right_arm'
    right_arm_action.joint_commands = np.array(act_ee_pose_right)
    right_arm_action.pose_command = None
    right_arm_action.duration = 0.0
    actions.actions.append(right_arm_action)

    left_hand_action = ComponentAction()
    left_hand_action.name = 'left_hand'
    left_hand_action.joint_commands = np.array([act_gripper_left])
    left_hand_action.pose_command = None
    left_hand_action.duration = 0.0
    actions.actions.append(left_hand_action)

    right_hand_action = ComponentAction()
    right_hand_action.name = 'right_hand'
    right_hand_action.joint_commands = np.array([act_gripper_right])
    right_hand_action.pose_command = None
    right_hand_action.duration = 0.0
    actions.actions.append(right_hand_action)

    actions.timestamp = rospy.Time.now().to_sec()
    return actions


def deploy_policy(bag_path, rate, use_tactile=False, end_sec=None,
                  use_rosbag_obs=False, use_rosbag_act=False, use_force_control=False):

    if bag_path != '':
        from data_pipeline.tools.robot_rosbag_decoder import RobotRosbagDecoder
        decoder = RobotRosbagDecoder()
        decoder.decode(bag_path, insert_previous_data=True)
        rosbag_count = 0

    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()

    client = RobotRosClient(frame_id='robot_ros_client')  # client = None
    rate = rospy.Rate(rate)

    while True:
        if rospy.is_shutdown():
            break

        if not use_rosbag_obs:
            # use realtime obs
            obs = client.get_observation(display_image=False)
        else:
            # use obs from rosbag
            rosbag_count = round((time.time() - START_TIME) * 1000 / 20.0)
            obs = decoder.observations[rosbag_count]

        action_chunk = infer(obs, policy_client, use_tactile)

        for act_i in range(25):  # 20 or action_chunk.shape[0] (50)

            if use_rosbag_act:
                rosbag_count = round((time.time() - START_TIME) * 1000 / 20.0)
                actions = decoder.actions[rosbag_count]
            else:
                actions = generate_action(action_chunk[act_i])

            # print('debug mode: not sending actions to the robot')
            if client is not None:
                client.send_action(actions)

            rate.sleep()

            if bag_path != '':
                print('rosbag_count', rosbag_count)

        if obs is not None:
            print(obs)
        rate.sleep()
    print('deploy policy done.')


def policy_infer_thread(client, decoder, infer_rate, act_rate, use_rosbag_act=False, use_tactile=False):
    rosbag_count = 0
    cur_obs_t = time.time()
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()

    global _G_ACTION_LIST
    while True:
        loop_start = time.time()
        if rospy.is_shutdown():
            break

        if client is not None:
            # use realtime obs
            obs = client.get_observation(display_image=False)
            # print('obs timestamp', time.time())
        elif decoder is not None:
            # use obs from rosbag
            rosbag_count = round((time.time() - START_TIME) * 1000 / 20.0)
            obs = decoder.observations[rosbag_count]
        else:
            print('Can not get observation without both client and decoder')

        # print(f'obs: {time.time() - cur_obs_t}')

        cur_obs_t = time.time()

        action_chunk = None
        # infer_start = time.time()
        if use_rosbag_act and decoder is not None:
            rosbag_count = round((time.time() - START_TIME) * 1000 / 20.0)
            action_chunk = [decoder.actions[rosbag_count]]
        else:
            action_chunk = infer(obs, policy_client, use_tactile).tolist()
        # print(f'infer: {time.time() - infer_start}')
        lock.acquire()
        cur_len = len(_G_ACTION_LIST)
        if cur_len == 0:
            _G_ACTION_LIST = action_chunk.copy()
        else:
            # print(f'infer - obs: {time.time() - cur_obs_t}')
            removed_len = math.ceil((time.time() - cur_obs_t) * act_rate)
            print('action chunk removed_len', removed_len)

            if removed_len < len(action_chunk):
                _G_ACTION_LIST = action_chunk[removed_len:]
            else:
                print(f'Predicted action length {len(action_chunk)} too less')

        lock.release()
        if decoder is not None:
            print('rosbag_count', rosbag_count)
        sleep_sec = (1 / infer_rate) - (time.time() - loop_start)
        if sleep_sec > 0:
            time.sleep(sleep_sec)
            print(f'sleep: {sleep_sec}')
        print(f'loop: {time.time() - loop_start}')
    print('deploy policy done.')


def send_action_thread(client, rate, use_force_control=False):
    global _G_ACTION_LIST
    ros_rate = rospy.Rate(rate)
    while not rospy.is_shutdown():
        ros_rate.sleep()
        cur_action = None
        lock.acquire()
        if len(_G_ACTION_LIST) != 0:
            cur_action = _G_ACTION_LIST.pop(0)
        lock.release()
        if cur_action is None:
            print('Empty action list')
            continue

        actions = RobotAction()
        if isinstance(cur_action, RobotAction):
            actions = cur_action
            actions.timestamp = rospy.Time.now().to_sec()
        else:
            actions = generate_action(cur_action)

        if client is not None:
            client.send_action(actions)
            pass

    print('send action done.')


if __name__ == '__main__':
    rospy.init_node('robot_ros_client_example', anonymous=True)
    rospy.loginfo("Ros node has been initialized")
    parser = argparse.ArgumentParser(description='Parameters from command line')
    # parser.add_argument('--bag_path', type=str, help='ros bag path', default='20250721-135113.bag')
    parser.add_argument('--bag_path', type=str, help='ros bag path', default='')
    parser.add_argument('--rate', type=float, help='action excution rate', default=10)  # 26, 28 or 27
    parser.add_argument('--parallel', type=bool, help='infer and execute on parallel', default=False)
    _ARGS, _ = parser.parse_known_args()
    if _ARGS.parallel:
        print('parallel inference')
        client = RobotRosClient(frame_id='robot_ros_client')  # client = None
        DECODER = None
        if _ARGS.bag_path != '':
            from data_pipeline.tools.robot_rosbag_decoder import RobotRosbagDecoder
            DECODER = RobotRosbagDecoder()
            DECODER.decode(_ARGS.bag_path, insert_previous_data=True)

        lock = threading.Lock()
        action_thread = threading.Thread(target=send_action_thread, args=(client, _ARGS.rate, False))
        action_thread.daemon = True
        action_thread.start()

        policy_thread = threading.Thread(target=policy_infer_thread, args=(
            client, DECODER, 4.0, _ARGS.rate))  # 4 is model infer frequence
        policy_thread.daemon = True
        policy_thread.start()

        rospy.spin()
        action_thread.join()
        policy_thread.join()
    else:
        print('serial inference')
        deploy_policy(_ARGS.bag_path, _ARGS.rate, use_tactile=False, use_force_control=False,
                      use_rosbag_obs=False, use_rosbag_act=False)
