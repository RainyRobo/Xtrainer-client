#!/usr/bin/env python3
import numpy as np
import rospy
import cv2
import data_msgs.msg as msg_type
from data_pipeline.tools.robot_message_conversion import (ObservationConversion,
                                                          ActionConversion,
                                                          TactileConversion)
from data_pipeline.tools.robot_action import ComponentAction, RobotAction


class RobotRosClient():
    def __init__(self, wait_first_msg=True, frame_id='default'):
        self.frame_id = frame_id
        self.action_pub = rospy.Publisher('/robot/actions', msg_type.RobotAction, queue_size=100)
        self.obs_sub = rospy.Subscriber('/robot/observations', msg_type.RobotObservation,
                                        self.observation_callback, queue_size=10)
        self.obs_msg = None
        if wait_first_msg:
            while self.obs_msg is None and not rospy.is_shutdown():
                rospy.Rate(0.5).sleep()
                rospy.loginfo_throttle(5, "Waiting for first message...")
            rospy.loginfo('Received robot observation')

    def observation_callback(self, msg):
        if self.obs_msg is None:
            self.obs_msg = msg
        self.obs_msg.header = msg.header
        self.obs_msg.component_observations = msg.component_observations
        if len(self.obs_msg.chain_images.chain_images) <= len(msg.chain_images.chain_images):
            self.obs_msg.chain_images = msg.chain_images
        if len(self.obs_msg.camera_images.images) <= len(msg.camera_images.images):
            self.obs_msg.camera_images = msg.camera_images
        self.obs_msg.tactiles = msg.tactiles

    def get_observation(self, display_image=False):
        if self.obs_msg is None:
            rospy.logwarn('Robot observation is not received')
            return None
        obs = ObservationConversion.from_robot_message(self.obs_msg)
        for tactile in self.obs_msg.tactiles:
            obs.tactiles.append(TactileConversion.from_robot_message(tactile))
        if display_image:
            for cam_img in obs.camera_images:
                if cam_img.image is None:
                    continue
                cv2.imshow(f'Camera images {cam_img.frame_id}', cam_img.image)
            for chain_img in obs.chain_images:
                if chain_img.color_image is None:
                    continue
                cv2.imshow(f'Chain images {chain_img.frame_id}', chain_img.color_image)
            cv2.waitKey(1)
        return obs

    def send_action(self, action):
        msg = ActionConversion.to_robot_message(action, self.frame_id)
        msg.header.stamp = rospy.Time.now()
        self.action_pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node('robot_ros_client_test', anonymous=True)
    wrapper = RobotRosClient('test')
    observations = wrapper.get_observation()
    rospy.loginfo(observations)
    actions = RobotAction()
    for obs in observations.observations:
        action = ComponentAction()
        action.name = obs.name
        action.joint_commands = np.zeros(obs.get_dof())
        action.joint_commands[0] += 1.0
        action.duration = 2.0
        actions.actions.append(action)
    actions.timestamp = rospy.Time.now().to_sec()
    wrapper.send_action(actions)
    for act in actions.actions:
        act.joint_commands[0] -= 1.0
        act.duration = 1.0
    wrapper.send_action(actions)
    rospy.spin()
