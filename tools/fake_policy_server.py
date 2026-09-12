#!/usr/bin/env python3
"""A stand-in StarVLA policy server that only twitches the grippers.

Speaks the real msgpack/websocket protocol, so ``inference_xtrainer.py`` runs
unmodified against it and the whole on-robot path -- observation capture,
handshake, action assembly, 100 Hz publishing -- is exercised for real. The
only thing that is fake is the policy: arm poses are echoed back unchanged so
the arms hold still, and the grippers step by a small amount.

    python3 tools/fake_policy_server.py --delta 0.05
    python3 inference_xtrainer.py --url ws://127.0.0.1:8000
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401

import argparse
import functools

import numpy as np
import websockets.exceptions
import websockets.sync.server

from vla.third_party.openpi_client import msgpack_numpy
from vla.xtrainer2_contract import (
    ACTION_DIM,
    CAMERA_ORDER,
    LEFT_GRIPPER,
    RIGHT_GRIPPER,
    STATE_DIM,
)

GRIPPER_INDEX = {'left': LEFT_GRIPPER, 'right': RIGHT_GRIPPER}

# This is watched live while the robot moves, so never sit in a pipe buffer.
log = functools.partial(print, flush=True)


def build_metadata(chunk):
    """The handshake ``check_server_metadata`` validates on the client side."""
    return {
        'action_chunk_size': chunk,
        'camera_order': [f'video.{camera}' for camera in CAMERA_ORDER],
        'state_dim': STATE_DIM,
        'action_dim': ACTION_DIM,
        'robot_quaternion_format': 'xyzw',
        'model_quaternion_format': 'wxyz',
        'server_normalizes_state': True,
        'image_format': 'HWC_RGB_uint8',
        'note': 'fake policy: holds arm poses, ramps grippers only',
    }


def describe_request(example):
    images = example.get('image') or []
    shapes = {f'{img.shape[0]}x{img.shape[1]}x{img.shape[2]}' for img in images
              if hasattr(img, 'shape')}
    return f'{len(images)} views {sorted(shapes)} lang={example.get("lang")!r}'


def check_state(state):
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape != (STATE_DIM,):
        raise ValueError(f'expected a {STATE_DIM}-D state; got {state.shape}')
    if not np.all(np.isfinite(state)):
        raise ValueError('state contains NaN or infinity')
    return state


def gripper_chunk(state, chunk, delta, hands, direction):
    """Hold both arm poses; ramp the selected grippers by ``delta``.

    The ramp is spread over the chunk so the robot sees a smooth trajectory
    rather than a step, which is what a real chunk looks like.
    """
    actions = np.tile(state, (chunk, 1)).astype(np.float32)
    ramp = np.linspace(1.0 / chunk, 1.0, chunk, dtype=np.float32)

    targets = {}
    for hand in hands:
        index = GRIPPER_INDEX[hand]
        start = float(state[index])
        target = float(np.clip(start + direction * delta, 0.0, 1.0))
        actions[:, index] = start + (target - start) * ramp
        targets[hand] = (start, target)
    return actions, targets


class FakePolicyServer:
    def __init__(self, host, port, chunk, delta, hands, oscillate):
        self.chunk = chunk
        self.delta = delta
        self.hands = hands
        self.oscillate = oscillate
        self.direction = -1.0  # close first; grippers usually start open
        self.requests = 0

        self._packer = msgpack_numpy.Packer()
        self._metadata = build_metadata(chunk)
        self._server = websockets.sync.server.serve(
            self._handle, host, port, compression=None, max_size=None)
        self.port = self._server.socket.getsockname()[1]

    def serve_forever(self):
        log(f'[fake] listening on ws://{self._server.socket.getsockname()[0]}:{self.port}')
        log(f'[fake] chunk={self.chunk} delta={self.delta} hands={",".join(self.hands)}')
        self._server.serve_forever()

    def _handle(self, connection):
        log('[fake] client connected; sending metadata')
        try:
            connection.send(self._packer.pack(self._metadata))
            for message in connection:
                try:
                    response = self._respond(msgpack_numpy.unpackb(message))
                except Exception as error:  # report instead of dropping the socket
                    log(f'[fake] rejecting request: {error}')
                    response = {
                        'status': 'error', 'ok': False, 'type': 'inference_result',
                        'error': {'message': str(error)},
                    }
                connection.send(self._packer.pack(response))
        except websockets.exceptions.ConnectionClosed:
            # Ctrl-C on the client is the normal way to end a rehearsal.
            pass
        log('[fake] client disconnected')

    def _respond(self, request):
        example = request['examples'][0]
        state = check_state(example['state'])
        actions, targets = gripper_chunk(
            state, self.chunk, self.delta, self.hands, self.direction)

        self.requests += 1
        moves = ' '.join(f'{hand}:{start:.3f}->{target:.3f}'
                         for hand, (start, target) in targets.items())
        log(f'[fake] #{self.requests} {describe_request(example)} | {moves}')
        if self.oscillate:
            self.direction = -self.direction

        return {
            'status': 'ok',
            'ok': True,
            'type': 'inference_result',
            'request_id': request.get('request_id', 'fake'),
            'data': {'actions': actions[None, ...]},
        }


def build_argparser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--chunk', type=int, default=16,
                        help='steps returned per inference')
    parser.add_argument('--delta', type=float, default=0.05,
                        help='gripper travel per chunk, in 0..1 units')
    parser.add_argument('--hands', default='left',
                        choices=['left', 'right', 'both'])
    parser.add_argument('--no-oscillate', action='store_true',
                        help='keep moving one way instead of alternating')
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    if args.chunk < 1:
        raise ValueError('--chunk must be at least 1')
    if not 0.0 < args.delta <= 1.0:
        raise ValueError('--delta must be within (0, 1]')

    hands = ('left', 'right') if args.hands == 'both' else (args.hands,)
    server = FakePolicyServer(args.host, args.port, args.chunk, args.delta,
                              hands, not args.no_oscillate)
    server.serve_forever()
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
