#!/usr/bin/env python3
"""Websocket bridge to the StarVLA policy server.

Kept free of ROS so the request/response path can be exercised off the robot;
``inference_xtrainer.py`` adds the control loop around it.
"""

import time

import cv2
import numpy as np

from vla.third_party.openpi_client import websocket_client_policy as _websocket_client_policy
from vla.xtrainer2_contract import (
    ACTION_DIM,
    CAMERA_ORDER,
    DEFAULT_INSTRUCTION,
    IMAGE_SIZE,
    build_state,
    check_server_metadata,
    resolve_camera_indices,
    split_action,
)


class PolicyBridge:
    """Turns a ROS observation into a request and a response into commands."""

    def __init__(self, url, instruction=DEFAULT_INSTRUCTION, unnorm_key=None,
                 camera_overrides=None, client=None, connect_timeout=None):
        self.instruction = instruction
        self.unnorm_key = unnorm_key
        self.camera_overrides = camera_overrides or {}
        self._camera_indices = None

        self.client = client or _websocket_client_policy.WebsocketClientPolicy(
            url, connect_timeout=connect_timeout)
        self.metadata = self.client.get_server_metadata()
        check_server_metadata(self.metadata)
        self.action_chunk_size = int(self.metadata.get("action_chunk_size", 0))
        if self.action_chunk_size <= 0:
            raise RuntimeError(f"server did not report action_chunk_size: {self.metadata}")
        print(f'[policy] connected to {url}')
        print(f'[policy] action_chunk_size={self.action_chunk_size} '
              f'instruction="{self.instruction}"')

    def camera_indices(self, obs):
        frame_ids = [img.frame_id for img in (obs.chain_images or [])]
        if len(frame_ids) < len(CAMERA_ORDER):
            raise ValueError(
                f'need {len(CAMERA_ORDER)} chain images, got {frame_ids}')
        if self._camera_indices is None:
            self._camera_indices = resolve_camera_indices(frame_ids, self.camera_overrides)
            print(f'[policy] chain images {frame_ids} -> {self._camera_indices}')
        return self._camera_indices

    def build_example(self, obs):
        """One request example: three views, the instruction, the raw state."""
        indices = self.camera_indices(obs)
        images = []
        for camera in CAMERA_ORDER:
            color = obs.chain_images[indices[camera]].color_image
            if color is None:
                raise RuntimeError(f'{camera} has no color image in this observation')
            # cv_bridge decodes to BGR; the model was trained on RGB frames.
            resized = cv2.resize(color, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            images.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

        state = build_state(
            left_pose=obs.get_observation(name='left_arm').multibody_pose,
            left_gripper=obs.get_observation(name='left_hand').multibody_state,
            right_pose=obs.get_observation(name='right_arm').multibody_pose,
            right_gripper=obs.get_observation(name='right_hand').multibody_state,
        )
        return {"image": images, "lang": self.instruction, "state": state}

    def infer(self, example):
        """Return one action chunk, shape ``(action_chunk_size, 16)``, in xyzw."""
        request = {"examples": [example]}
        if self.unnorm_key is not None:
            request["unnorm_key"] = self.unnorm_key

        start = time.time()
        response = self.client.infer(request)
        elapsed = time.time() - start

        if not response.get("ok", False):
            raise RuntimeError(f'policy server error: {response.get("error")}')
        actions = np.asarray(response["data"]["actions"][0], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
            raise RuntimeError(f'expected (T, {ACTION_DIM}) actions; got {actions.shape}')
        print(f'action_chunk {actions.shape} time {elapsed:.3f}s')
        return actions


def check_first_step(action, state, max_jump):
    """Reject a chunk whose first command teleports the arms.

    These are absolute end-effector poses at 10 Hz, so the first step of a chunk
    sits near the current pose. A large jump means the commands are not in the
    space the robot expects -- wrong pose frame, wrong quaternion order, or a
    checkpoint trained on another embodiment -- and must not reach the arms.
    """
    if max_jump <= 0:
        return
    for name, offset in (('left', 0), ('right', 8)):
        distance = float(np.linalg.norm(action[offset:offset + 3] - state[offset:offset + 3]))
        if distance > max_jump:
            raise RuntimeError(
                f'{name} arm: first commanded position is {distance:.3f} m from the current '
                f'pose (limit {max_jump:.3f} m). Refusing to send. '
                'Check --pose-frame (TorsoEE), the checkpoint and the camera mapping, '
                'or raise --max-first-step-jump if this is intended.'
            )


__all__ = ["PolicyBridge", "check_first_step", "split_action"]
