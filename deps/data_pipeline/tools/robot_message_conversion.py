#!/usr/bin/env python3
import numpy as np
from cv_bridge import CvBridge
import rospy
from geometry_msgs.msg import Pose
import data_msgs.msg as msg_type
from data_pipeline.tools.robot_observation import (ComponentObservation, RobotObservation,
                                                   ForceTorque, ChainImage, CameraImage,
                                                   Tactile, Event)
from data_pipeline.tools.robot_action import ComponentAction, RobotAction
from data_pipeline.tools.robot_config import JointConfig, ComponentConfig, RobotConfig

# Converts geometry_pose/Pose to numpy array in format of [x,y,z,qx,qy,qz,qw].


def from_pose_message(msg):
    return np.array([msg.position.x, msg.position.y, msg.position.z,
                     msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])


def to_pose_message(pose):
    msg = Pose()
    msg.position.x = pose[0]
    msg.position.y = pose[1]
    msg.position.z = pose[2]
    msg.orientation.x = pose[3]
    msg.orientation.y = pose[4]
    msg.orientation.z = pose[5]
    msg.orientation.w = pose[6]
    return msg


def from_wrench_message(wrench):
    return np.array([wrench.force.x, wrench.force.y, wrench.force.z,
                     wrench.torque.x, wrench.torque.y, wrench.torque.z])


class ObservationConversion:
    @staticmethod
    # `msg` defined as ComponentObservation.msg
    def from_component_message(msg):
        obs = ComponentObservation()
        obs.name = msg.header.frame_id
        for pose_msg in msg.track_poses:
            obs.track_poses.append(from_pose_message(pose_msg.pose))
        if msg.multibody_pose.header.frame_id != '':
            obs.multibody_pose = from_pose_message(msg.multibody_pose.pose)
        if len(msg.multibody_state.states) > 0:
            obs.multibody_state = np.array([state_msg.q for state_msg in msg.multibody_state.states])
        if len(msg.multibody_command.commands) > 0:
            obs.multibody_command = np.array([cmd_msg.values[0] for cmd_msg in msg.multibody_command.commands])
        if hasattr(msg, 'force_torques'):
            for ft_msg in msg.force_torques:
                cur_ft = ForceTorque()
                cur_ft.frame_id = ft_msg.header.frame_id
                cur_ft.data = from_wrench_message(ft_msg.wrench)
                obs.force_torques.append(cur_ft)
        return obs

    @staticmethod
    # `msg` defined as RobotObservation.msg
    def from_robot_message(msg):
        obs = RobotObservation()
        obs.timestamp = msg.header.stamp.to_sec()
        for component_obs in msg.component_observations:
            obs.observations.append(ObservationConversion.from_component_message(component_obs))

        bridge = CvBridge()
        for cam_img in msg.camera_images.images:
            img = CameraImage()
            img.frame_id = cam_img.header.frame_id
            img.timestamp = cam_img.header.stamp.to_sec()
            if len(cam_img.data) != 0:
                img.image = bridge.compressed_imgmsg_to_cv2(cam_img)
            obs.camera_images.append(img)

        for chain_img in msg.chain_images.chain_images:
            img = ChainImage()
            img.frame_id = chain_img.header.frame_id
            img.timestamp = chain_img.header.stamp.to_sec()
            if len(chain_img.color_image.data) != 0:
                img.color_image = bridge.compressed_imgmsg_to_cv2(chain_img.color_image)
            if len(chain_img.depth_image.data) != 0:
                img.depth_image = bridge.compressed_imgmsg_to_cv2(chain_img.depth_image)
            if len(chain_img.ir_image.data) != 0:
                img.ir_image = bridge.compressed_imgmsg_to_cv2(chain_img.ir_image)
            obs.chain_images.append(img)
        return obs


class ActionConversion:
    @staticmethod
    # `msg` defined as ComponentAction.msg
    def from_component_message(msg):
        act = ComponentAction()
        act.name = msg.header.frame_id
        if msg.pose_command.header.frame_id != '':
            act.pose_command = act.Pose(from_pose_message(msg.pose_command.pose),
                                        msg.pose_command.header.frame_id)
        if len(msg.joint_commands) > 0:
            act.joint_commands = np.array(msg.joint_commands)
        act.duration = msg.duration
        return act

    @staticmethod
    # `msg` defined as RobotAction.msg
    def from_robot_message(msg):
        act = RobotAction()
        act.timestamp = msg.header.stamp.to_sec()
        for component_act in msg.component_actions:
            cur_act = ActionConversion.from_component_message(component_act)
            if cur_act.is_valid():
                act.actions.append(ActionConversion.from_component_message(component_act))
        return act

    @staticmethod
    # `action` defined as ComponentAction
    def to_component_message(action):
        msg = msg_type.ComponentAction()
        msg.header.frame_id = action.name
        if action.pose_command is not None:
            msg.pose_command.header.frame_id = action.pose_command.frame
            msg.pose_command.pose = to_pose_message(action.pose_command.pose)
        if action.joint_commands is not None:
            msg.joint_commands = action.joint_commands.tolist()
        msg.duration = action.duration
        return msg

    @staticmethod
    # `action` defined as RobotAction
    def to_robot_message(action, frame_id=''):
        msg = msg_type.RobotAction()
        if action.timestamp is not None:
            msg.header.stamp = rospy.Time.from_sec(action.timestamp)
        msg.header.frame_id = frame_id
        for act in action.actions:
            msg.component_actions.append(ActionConversion.to_component_message(act))
            msg.component_actions[-1].header.stamp = msg.header.stamp
        return msg


class ConfigConversion:
    @staticmethod
    # `msg` defined as JointConfig.msg
    def from_joint_message(msg):
        config = JointConfig()
        config.name = msg.name
        config.lower_position = msg.lower_position
        config.upper_position = msg.upper_position
        config.supported_operation_modes = msg.supported_operation_modes
        return config

    @staticmethod
    # `msg` defined as MultibodyConfig.msg
    def from_component_message(msg):
        config = ComponentConfig()
        config.operation_mode_names = msg.operation_mode_names
        for j_config in msg.configs:
            config.joint_configs.append(ConfigConversion.from_joint_message(j_config))
        return config

    @staticmethod
    # `msg` defined as RobotConfig.msg
    def from_robot_message(msg):
        config = RobotConfig()
        config.name = msg.header.frame_id
        for component_config in msg.configs:
            config.configs.append(ConfigConversion.from_component_message(component_config))
        return config


class TactileConversion:
    @staticmethod
    def from_robot_message(msg):
        tactile = Tactile()
        tactile.name = msg.layout.dim[0].label
        tactile.data = np.array(msg.data)
        return tactile


class EventConversion:
    @staticmethod
    def from_robot_message(msg):
        if msg.event_type == '' and msg.event_detail == '':
            return None
        event = Event()
        event.type = msg.event_type
        event.detail = msg.event_detail
        return event
