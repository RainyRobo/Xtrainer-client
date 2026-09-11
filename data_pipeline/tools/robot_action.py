#!/usr/bin/env python3
import collections


class ComponentAction():
    # `pose` is numpy array in format of [x,y,z,qx,qy,qz,qw] in robot torso frame.
    # `frame` is string of TorsoTool or TorsoEe.
    Pose = collections.namedtuple('Pose', ['pose', 'frame'])

    def __init__(self):
        # string of `xxx_arm/hand_left/right`
        self.name = None
        # In `Pose` format.
        self.pose_command = None
        # 1xDoF numpy array
        self.joint_commands = None
        self.duration = 0.0

    def __str__(self):
        return (f'name:{self.name}; duration:{self.duration}\n'
                f'pose_command:{self.pose_command}\n'
                f'joint_commands:{self.joint_commands}')

    def get_dof(self):
        if self.name is None:
            return None
        return len(self.joint_commands)

    # `type`` is arm or hand.
    def is_type(self, type):
        if self.name is None:
            return False
        return type in self.name

    def is_valid(self):
        return (self.name is not None
                and (self.pose_command is not None or self.joint_commands is not None))


class RobotAction():
    def __init__(self):
        # Generated timestamp in second
        self.timestamp = None
        # List of `ComponentAction`
        self.actions = []

    def __str__(self):
        return (f'timestamp:{self.timestamp}; num of actions:{len(self.actions)}')

    # `kwargs` supports the following keys:
    #   - name
    #   - type and location
    # Returns `ComponentAction` if the key match.
    def get_action(self, **kwargs):
        if 'name' in kwargs.keys():
            for act in self.actions:
                if kwargs['name'] == act.name:
                    return act
        elif 'type' in kwargs.keys() and 'location' in kwargs.keys():
            for act in self.actions:
                if act.is_type(kwargs['type']) and kwargs['location'] in act.name:
                    return act
        else:
            print(f'Unknown args {kwargs}')
        return None
