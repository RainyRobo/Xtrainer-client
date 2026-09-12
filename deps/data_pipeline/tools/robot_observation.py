#!/usr/bin/env python3

class ForceTorque():
    def __init__(self):
        self.frame_id = None
        # List of numpy array in format of [force_x, force_y, force_z, torque_x, torque_y, torque_z]
        self.data = []

    def __str__(self):
        return (f'frame_id:{self.frame_id}; data:{self.data}')


class ComponentObservation():
    def __init__(self):
        # string of `xxx_arm/hand_left/right`
        self.name = None
        # List of numpy array in format of [x,y,z,qx,qy,qz,qw]
        self.track_poses = []
        # List of `ForceTorque`
        self.force_torques = []
        # Numpy array in format of [x,y,z,qx,qy,qz,qw] of component last link in robot torso frame.
        self.multibody_pose = None
        # 1xDoF numpy array
        self.multibody_state = None
        # 1xDoF numpy array
        self.multibody_command = None

    def __str__(self):
        return (f'name:{self.name}\n'
                f'track_poses:{self.track_poses}\n'
                f'force_torques:{self.force_torques}\n'
                f'multibody_pose:{self.multibody_pose}\n'
                f'multibody_state:{self.multibody_state}\n'
                f'multibody_command:{self.multibody_command}')

    def get_dof(self):
        if self.name is None:
            return None
        return len(self.multibody_state)

    # `type`` is arm or hand.
    def is_type(self, type):
        if self.name is None:
            return False
        return type in self.name


class CameraImage():
    def __init__(self):
        self.frame_id = None
        # Generated timestamp in second
        self.timestamp = None
        # In cv2 format data
        self.image = None


class ChainImage():
    def __init__(self):
        self.frame_id = None
        # Generated timestamp in second
        self.timestamp = None
        # In cv2 format
        self.color_image = None
        self.depth_image = None
        self.ir_image = None


class Tactile():
    def __init__(self):
        self.name = None
        # 1xDoF numpy array
        self.data = None


class Event():
    def __init__(self):
        self.type = None
        self.detail = None

    def __str__(self):
        return (f'type:{self.type}\n'
                f'detail:{self.detail}')


class RobotObservation():
    def __init__(self):
        # Generated timestamp in second
        self.timestamp = None
        # List of `ComponentObservation`
        self.observations = []
        # List of `CameraImage`
        self.camera_images = []
        # List of `ChainImages`
        self.chain_images = []
        # List of `Tactile`
        self.tactiles = []
        # `Event` type if any event happens at current timestamp.
        self.event = None

    def __str__(self):
        return (f'timestamp:{self.timestamp}; num of observations:{len(self.observations)};'
                f' num of camera images:{len(self.camera_images)};'
                f' num of chain images:{len(self.chain_images)};'
                f' num of tactiles:{len(self.tactiles)};'
                f' event: {self.event}')

    # `kwargs` supports the following keys:
    #   - name
    #   - type and location
    # Returns `ComponentAction` if the key match.
    # Returns `ComponentObservation` if name matches.
    def get_observation(self, **kwargs):
        if 'name' in kwargs.keys():
            for obs in self.observations:
                if kwargs['name'] == obs.name:
                    return obs
        elif 'type' in kwargs.keys() and 'location' in kwargs.keys():
            for obs in self.observations:
                if obs.is_type(kwargs['type']) and kwargs['location'] in obs.name:
                    return obs
        else:
            print(f'Unknown args {kwargs}')
        return None
