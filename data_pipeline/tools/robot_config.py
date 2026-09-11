#!/usr/bin/env python3

class JointConfig():
    def __init__(self):
        self.name = None
        self.lower_position = None
        self.upper_position = None
        self.supported_operation_modes = []

    def __str__(self):
        return (f'name:{self.name}: lower_position:{self.lower_position}'
                f'; upper_position:{self.upper_position}'
                f'; supported_operation_modes:{self.supported_operation_modes}')


class ComponentConfig():
    def __init__(self):
        self.operation_mode_names = []
        # List of `JointConfig`
        self.joint_configs = []

    def __str__(self):
        return "operation_mode_names:{}\n{}".format(self.operation_mode_names, '\n'.join(map(str, self.joint_configs)))


class RobotConfig():
    def __init__(self):
        self.name = None
        # List of `ComponentConfig`
        self.configs = []

    def __str__(self):
        return "name:{}\n{}".format(self.name, '\n'.join(map(str, self.configs)))
