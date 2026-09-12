#!/usr/bin/env python3
"""Pre-flight probe: look at one observation (mock, rosbag, or live robot).

Does not publish actions. With ``--url`` it also handshakes the policy server
and runs a single inference so you can see metadata, latency, and how far the
first commanded poses sit from the current ones.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401

import argparse
import json
import shlex
import subprocess
import time

import numpy as np

# Same tree the robot docker mounts as /data. Host home is tmpfs inside that container.
LIVE_CLIENT = os.environ.get('XTRAINER_LIVE_CLIENT', '/data/arianliu/client')
ROBOT_DOCKER = os.environ.get('XTRAINER_DOCKER', 'master')

from vla.xtrainer2_contract import (
    CAMERA_ORDER,
    DEFAULT_INSTRUCTION,
    IMAGE_SIZE,
    build_state,
    resolve_camera_indices,
    split_action,
)


class _ChainImage:
    def __init__(self, frame_id, color_image):
        self.frame_id = frame_id
        self.color_image = color_image
        self.timestamp = None
        self.depth_image = None


class _Component:
    def __init__(self, name, multibody_pose=None, multibody_state=None):
        self.name = name
        self.multibody_pose = multibody_pose
        self.multibody_state = multibody_state


class MockObservation:
    """The fields ``PolicyBridge.build_example`` reads, plus a component list."""

    def __init__(self):
        rng = np.random.default_rng(0)
        self.timestamp = 0.0
        self.chain_images = [
            _ChainImage('right_wrist_camera', _bgr_frame(30, rng)),
            _ChainImage('head_camera', _bgr_frame(10, rng)),
            _ChainImage('left_wrist_camera', _bgr_frame(20, rng)),
        ]
        self.camera_images = []
        self.observations = [
            _Component('left_arm',
                       multibody_pose=np.array([0.25, 0.18, 0.12, 0.0, 0.0, 0.0, 1.0])),
            _Component('right_arm',
                       multibody_pose=np.array([0.25, -0.18, 0.12, 0.0, 0.0, 0.0, 1.0])),
            _Component('left_hand', multibody_state=np.array([0.04])),
            _Component('right_hand', multibody_state=np.array([0.04])),
        ]

    def get_observation(self, name=None, **kwargs):
        key = name or kwargs.get('name')
        for component in self.observations:
            if component.name == key:
                return component
        return None


def _bgr_frame(grey, rng):
    image = np.full((480, 848, 3), grey, dtype=np.uint8)
    image[:8, :8] = rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
    return image


def _fmt(values, digits=3):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return '[' + ', '.join(f'{v:.{digits}f}' for v in values) + ']'


def _pose_line(pose):
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose.size != 7:
        return f'{_fmt(pose)}  (expected 7-D xyz+xyzw, got {pose.size})'
    quat = pose[3:7]
    return (f'xyz={_fmt(pose[:3])}  xyzw={_fmt(quat)}  '
            f'|q|={float(np.linalg.norm(quat)):.4f}')


def _image_line(image):
    if image is None:
        return 'missing'
    array = np.asarray(image)
    return f'shape={tuple(array.shape)} dtype={array.dtype} mean={array.mean():.1f}'


def observation_state(obs):
    left_arm = obs.get_observation(name='left_arm')
    right_arm = obs.get_observation(name='right_arm')
    left_hand = obs.get_observation(name='left_hand')
    right_hand = obs.get_observation(name='right_hand')
    missing = [name for name, component in (
        ('left_arm', left_arm), ('right_arm', right_arm),
        ('left_hand', left_hand), ('right_hand', right_hand),
    ) if component is None]
    if missing:
        raise RuntimeError(f'observation is missing components: {missing}')
    return build_state(
        left_pose=left_arm.multibody_pose,
        left_gripper=left_hand.multibody_state,
        right_pose=right_arm.multibody_pose,
        right_gripper=right_hand.multibody_state,
    )


def summarize_observation(obs, camera_overrides=None):
    """Human-readable dump of the fields that later go to the server."""
    lines = []
    timestamp = getattr(obs, 'timestamp', None)
    lines.append(f'timestamp: {timestamp}')

    components = list(getattr(obs, 'observations', []) or [])
    if not components:
        for name in ('left_arm', 'right_arm', 'left_hand', 'right_hand'):
            component = obs.get_observation(name=name)
            if component is not None:
                components.append(component)
    lines.append(f'components ({len(components)}):')
    for component in components:
        name = getattr(component, 'name', '?')
        pose = getattr(component, 'multibody_pose', None)
        joints = getattr(component, 'multibody_state', None)
        bits = [f'  {name}']
        if pose is not None:
            bits.append(f'    pose  {_pose_line(pose)}')
        if joints is not None:
            joints = np.asarray(joints).reshape(-1)
            bits.append(f'    state {_fmt(joints)}  dof={joints.size}')
        lines.extend(bits)

    chain_images = list(getattr(obs, 'chain_images', []) or [])
    frame_ids = [getattr(image, 'frame_id', None) for image in chain_images]
    lines.append(f'chain images ({len(chain_images)}): {frame_ids}')
    for image in chain_images:
        lines.append(f'  {image.frame_id}: {_image_line(getattr(image, "color_image", None))}')

    mapping_error = None
    mapping = None
    try:
        mapping = resolve_camera_indices(frame_ids, camera_overrides)
        lines.append(f'camera mapping (training order {list(CAMERA_ORDER)}): {mapping}')
        for camera in CAMERA_ORDER:
            index = mapping[camera]
            image = chain_images[index]
            lines.append(f'  {camera} <- [{index}] {image.frame_id}  '
                         f'{_image_line(image.color_image)}')
    except (ValueError, IndexError) as exc:
        mapping_error = str(exc)
        lines.append(f'camera mapping FAILED: {exc}')

    state = None
    try:
        state = observation_state(obs)
        left_pose, left_grip, right_pose, right_grip = split_action(state)
        lines.append(f'16-D wire state: {_fmt(state)}')
        lines.append(f'  left  {_pose_line(left_pose)}  gripper={left_grip:.4f}')
        lines.append(f'  right {_pose_line(right_pose)}  gripper={right_grip:.4f}')
    except (RuntimeError, ValueError) as exc:
        lines.append(f'16-D wire state FAILED: {exc}')

    return {
        'lines': lines,
        'state': state,
        'mapping': mapping,
        'mapping_error': mapping_error,
        'frame_ids': frame_ids,
    }


def summarize_actions(actions, state, max_jump):
    actions = np.asarray(actions, dtype=np.float32)
    first = actions[0]
    lines = [f'action chunk shape {actions.shape}  (T, 16)']
    for name, offset in (('left', 0), ('right', 8)):
        jump = float(np.linalg.norm(first[offset:offset + 3] - state[offset:offset + 3]))
        span = float(np.linalg.norm(actions[-1, offset:offset + 3] - first[offset:offset + 3]))
        grip = first[offset + 7]
        flag = ''
        if max_jump > 0 and jump > max_jump:
            flag = f'  EXCEEDS --max-first-step-jump {max_jump:.3f} m'
        lines.append(
            f'  {name}: first xyz jump {jump:.4f} m, chunk xyz travel {span:.4f} m, '
            f'first gripper {grip:.4f}{flag}'
        )
    left_pose, left_grip, right_pose, right_grip = split_action(first)
    lines.append(f'  first left  {_pose_line(left_pose)}  gripper={left_grip:.4f}')
    lines.append(f'  first right {_pose_line(right_pose)}  gripper={right_grip:.4f}')
    return lines


def _rospy_is_usable():
    try:
        import roslib  # noqa: F401
        import rospy  # noqa: F401
    except ImportError:
        return False
    return True


def _in_robot_docker():
    return os.path.exists('/.dockerenv')


def _sync_client_to_data():
    """Copy this checkout to /data so the robot docker can see it."""
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.realpath(src) == os.path.realpath(LIVE_CLIENT):
        return LIVE_CLIENT
    os.makedirs(LIVE_CLIENT, exist_ok=True)
    cmd = [
        'rsync', '-a',
        '--exclude', '.git',
        '--exclude', '__pycache__',
        '--exclude', '.pytest_cache',
        '--exclude', '.live_ros_msgs',
        src + '/', LIVE_CLIENT + '/',
    ]
    print(f'[probe] sync {src} -> {LIVE_CLIENT}', flush=True)
    subprocess.check_call(cmd)
    return LIVE_CLIENT


def _reexec_in_ros1_docker(argv):
    """Run in the already-running robot docker (synced with the arms), not a new image."""
    if _in_robot_docker():
        raise ImportError(
            'inside docker but rospy is missing; from /data/robotics.install run: '
            'source /opt/ros/noetic/setup.bash && source env.sh'
        )

    live = _sync_client_to_data()
    inspect = subprocess.run(
        ['docker', 'inspect', '-f', '{{.State.Running}}', ROBOT_DOCKER],
        capture_output=True, text=True,
    )
    if inspect.returncode != 0 or inspect.stdout.strip() != 'true':
        raise RuntimeError(
            f'robot docker {ROBOT_DOCKER!r} is not running. On the robot:\n'
            f'  cd /data/robotics.install && bash scripts/run_docker.sh\n'
            f'  cd {live} && python3 tools/probe_xtrainer.py --robot'
        )

    inner = (
        'source /opt/ros/noetic/setup.bash && '
        'cd /data/robotics.install && source env.sh && '
        'cd ' + shlex.quote(live) + ' && '
        'python3 tools/probe_xtrainer.py'
    )
    if argv:
        inner += ' ' + ' '.join(shlex.quote(a) for a in argv)
    cmd = [
        'docker', 'exec',
        '-e', 'XTRAINER_PROBE_IN_DOCKER=1',
        '-e', 'ROS_MASTER_URI=http://localhost:11311',
        ROBOT_DOCKER, 'bash', '-lc', inner,
    ]
    cmd.insert(2, '-it' if sys.stdin.isatty() else '-i')
    print(f'[probe] running in docker {ROBOT_DOCKER} at {live}', flush=True)
    os.execvp(cmd[0], cmd)


def connect_live_robot(wait_timeout):
    if not _rospy_is_usable():
        if os.environ.get('XTRAINER_PROBE_IN_DOCKER') == '1':
            raise ImportError('ROS1 Noetic is missing rospy/roslib inside the container')
        _reexec_in_ros1_docker(sys.argv[1:])

    import rosgraph
    import rospy
    from data_pipeline.tools.robot_ros_client import RobotRosClient

    master = os.environ.get('ROS_MASTER_URI', 'http://localhost:11311')
    if not rosgraph.is_master_online():
        raise RuntimeError(
            f'ROS master is not running at {master}. '
            'Start the robot first, e.g. cd /data/robotics.install && ./run_collection.sh, '
            'then rerun python3 tools/probe_xtrainer.py --robot'
        )

    rospy.init_node('xtrainer2_probe', anonymous=True)
    client = RobotRosClient(wait_first_msg=False, frame_id='xtrainer2_probe')
    deadline = time.time() + wait_timeout
    # The first messages can arrive before the cameras publish, and the images
    # are what the policy needs; keep waiting until a frame carries all three.
    obs = None
    while time.time() < deadline and not rospy.is_shutdown():
        if client.obs_msg is not None:
            obs = client.get_observation(display_image=False)
            if obs is not None and len(obs.chain_images) >= len(CAMERA_ORDER):
                return client, obs
        rospy.sleep(0.2)

    if obs is not None:
        print(f'[probe] warning: only {len(obs.chain_images)} chain image(s) after '
              f'{wait_timeout:.0f}s; check /robot/status/*_orbbec', flush=True)
        return client, obs

    topics = sorted(name for name, _typ in rospy.get_published_topics())
    preview = ', '.join(topics[:20]) if topics else '(none)'
    import data_msgs.msg as msg_type
    md5 = getattr(msg_type.RobotObservation, '_md5sum', '?')
    raise RuntimeError(
        f'subscribed to /robot/observations but got no callbacks in {wait_timeout:.0f}s '
        f'(local RobotObservation md5={md5}). If the topic is listed below, the message '
        f'definition likely does not match the robot. Published topics: {preview}'
    )


def _hand_opening(obs, name):
    component = obs.get_observation(name=name)
    if component is None or component.multibody_state is None:
        raise RuntimeError(f'observation has no {name} gripper state')
    return float(np.asarray(component.multibody_state, dtype=np.float64).reshape(-1)[0])


def build_gripper_action(obs, left_opening, right_opening, pose_frame, duration, timestamp):
    """Keep current arm poses; command only the two grippers in 0..1."""
    from data_pipeline.tools.robot_action import ComponentAction, RobotAction

    actions = RobotAction()
    for name in ('left_arm', 'right_arm'):
        arm = obs.get_observation(name=name)
        if arm is None or arm.multibody_pose is None:
            raise RuntimeError(f'cannot hold {name}: missing multibody_pose')
        arm_action = ComponentAction()
        arm_action.name = name
        arm_action.pose_command = ComponentAction.Pose(
            np.asarray(arm.multibody_pose, dtype=np.float64), pose_frame)
        arm_action.joint_commands = None
        arm_action.duration = 0.0
        actions.actions.append(arm_action)

    for name, opening in (('left_hand', left_opening), ('right_hand', right_opening)):
        opening = float(np.clip(opening, 0.0, 1.0))
        hand_action = ComponentAction()
        hand_action.name = name
        hand_action.pose_command = None
        hand_action.joint_commands = np.array([opening], dtype=np.float64)
        hand_action.duration = duration
        actions.actions.append(hand_action)

    actions.timestamp = timestamp
    return actions


def _wait_for_action_subscriber(client, timeout=3.0):
    import rospy
    deadline = time.time() + timeout
    while client.action_pub.get_num_connections() == 0 and time.time() < deadline:
        rospy.sleep(0.1)
    n = client.action_pub.get_num_connections()
    if n == 0:
        raise RuntimeError(
            'nobody is subscribed to /robot/actions; the robot stack is not listening'
        )
    print(f'[probe] /robot/actions has {n} subscriber(s)', flush=True)


def check_robot_ready():
    """A subscriber on /robot/actions is not enough; the agent must be running.

    While /robot/status/agent is still initializing the arms accept nothing, so
    commands look like they were sent successfully and nothing moves.
    """
    import rospy
    from diagnostic_msgs.msg import DiagnosticStatus

    try:
        status = rospy.wait_for_message('/robot/status/agent', DiagnosticStatus, timeout=5.0)
    except rospy.ROSException:
        print('[probe] warning: no /robot/status/agent within 5s', flush=True)
        return
    if status.level != DiagnosticStatus.OK:
        raise RuntimeError(
            f'robot agent is not ready: level={status.level} name={status.name!r} '
            f'{status.message!r}. Long press the yellow button on each leader arm to '
            'start the agents, then rerun. Commands sent now are silently ignored.'
        )
    print(f'[probe] agent status ok ({status.name})', flush=True)


def run_gripper_cycle(client, obs, hands, close_to, open_to, duration, pose_frame, send):
    import rospy

    left = _hand_opening(obs, 'left_hand')
    right = _hand_opening(obs, 'right_hand')
    print(f'[probe] gripper now  left={left:.3f}  right={right:.3f}  (0=closed, 1=open)', flush=True)

    close_left = close_to if hands in ('both', 'left') else left
    close_right = close_to if hands in ('both', 'right') else right
    open_left = open_to if hands in ('both', 'left') else left
    open_right = open_to if hands in ('both', 'right') else right

    steps = (
        ('close', close_left, close_right),
        ('open', open_left, open_right),
    )
    if not send:
        for label, lval, rval in steps:
            print(f'[probe] would {label} grippers to left={lval:.3f} right={rval:.3f} '
                  f'(pass --send-gripper to publish)')
        return 0

    _wait_for_action_subscriber(client)
    check_robot_ready()
    for label, lval, rval in steps:
        action = build_gripper_action(
            obs, lval, rval, pose_frame, duration, rospy.Time.now().to_sec())
        print(f'[probe] sending {label}: left={lval:.3f} right={rval:.3f} duration={duration:.1f}s',
              flush=True)
        client.send_action(action)
        rospy.sleep(duration + 0.4)
        obs = client.get_observation(display_image=False) or obs
        print(
            f'[probe] gripper after {label}  '
            f'left={_hand_opening(obs, "left_hand"):.3f}  '
            f'right={_hand_opening(obs, "right_hand"):.3f}',
            flush=True,
        )
    return 0


def load_bag_observation(bag_path):
    from data_pipeline.tools.robot_rosbag_decoder import RobotRosbagDecoder

    decoder = RobotRosbagDecoder()
    decoder.decode(bag_path, insert_previous_data=True)
    if not decoder.observations:
        raise RuntimeError(f'no observations in {bag_path}')
    return decoder.observations[0]


def infer_once(url, obs, instruction, unnorm_key, camera_overrides,
               connect_timeout, max_jump):
    from vla.xtrainer2_policy import PolicyBridge, check_first_step

    bridge = PolicyBridge(
        url=url,
        instruction=instruction,
        unnorm_key=unnorm_key,
        camera_overrides=camera_overrides,
        connect_timeout=connect_timeout,
    )
    metadata = {key: bridge.metadata.get(key) for key in (
        'action_chunk_size', 'camera_order', 'state_dim', 'action_dim',
        'robot_quaternion_format', 'model_quaternion_format',
        'server_normalizes_state', 'image_format',
    )}
    example = bridge.build_example(obs)
    images = example['image']
    image_lines = [
        f'request images: {len(images)} x {IMAGE_SIZE[1]}x{IMAGE_SIZE[0]} RGB uint8'
    ]
    for camera, image in zip(CAMERA_ORDER, images):
        image_lines.append(f'  {camera}: {_image_line(image)}')

    actions = bridge.infer(example)
    jump_lines = summarize_actions(actions, example['state'], max_jump)
    jump_ok = True
    try:
        check_first_step(actions[0], example['state'], max_jump)
    except RuntimeError as exc:
        jump_ok = False
        jump_lines.append(f'first-step guard: {exc}')
    else:
        jump_lines.append('first-step guard: ok (would allow this chunk)')

    return metadata, image_lines, jump_lines, jump_ok


def build_argparser():
    parser = argparse.ArgumentParser(description='xtrainer2 pre-flight probe (no motion)')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--mock', action='store_true',
                        help='synthetic cameras and poses (default if no ROS source)')
    source.add_argument('--robot', action='store_true',
                        help='read one live observation from /robot/observations')
    source.add_argument('--bag-path', type=str, default='',
                        help='read the first observation from a rosbag/mcap')
    parser.add_argument('--url', type=str, default='',
                        help='optional policy server; handshake + one inference, no publish')
    parser.add_argument('--connect-timeout', type=float, default=15.0,
                        help='seconds to wait for the server before giving up')
    parser.add_argument('--instruction', type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument('--unnorm-key', type=str, default=None)
    parser.add_argument('--max-first-step-jump', type=float, default=0.15)
    parser.add_argument('--wait-timeout', type=float, default=20.0,
                        help='seconds to wait for /robot/observations')
    parser.add_argument('--send-gripper', action='store_true',
                        help='publish a close-then-open gripper cycle on /robot/actions')
    parser.add_argument('--gripper-cycle', action='store_true',
                        help='plan a close/open cycle (add --send-gripper to actually move)')
    parser.add_argument('--hands', type=str, default='both',
                        choices=['both', 'left', 'right'])
    parser.add_argument('--gripper-close', type=float, default=0.0,
                        help='closed command in 0..1 (0 is fully closed)')
    parser.add_argument('--gripper-open', type=float, default=1.0,
                        help='open command in 0..1 (1 is fully open)')
    parser.add_argument('--gripper-duration', type=float, default=1.5,
                        help='seconds allowed for each gripper move')
    parser.add_argument('--pose-frame', type=str, default='TorsoEe',
                        choices=['TorsoEe', 'TorsoTool'],
                        help='frame used to hold the arms while the grippers move')
    for camera in CAMERA_ORDER:
        parser.add_argument(f'--{camera.replace("_", "-")}', type=str, default=None)
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    camera_overrides = {
        camera: getattr(args, camera) for camera in CAMERA_ORDER
        if getattr(args, camera) is not None
    }

    client = None
    if args.robot:
        source = 'live robot /robot/observations'
        client, obs = connect_live_robot(args.wait_timeout)
    elif args.bag_path:
        source = f'rosbag {args.bag_path}'
        obs = load_bag_observation(args.bag_path)
    else:
        source = 'synthetic mock observation'
        obs = MockObservation()

    print(f'[probe] source: {source}')
    if not args.send_gripper:
        print('[probe] this script does not publish /robot/actions unless you pass --send-gripper')
    dump = summarize_observation(obs, camera_overrides)
    print('\n'.join(dump['lines']))

    if args.send_gripper or args.gripper_cycle:
        if client is None:
            print('[probe] gripper cycle needs --robot so it can publish /robot/actions')
            return 1
        return run_gripper_cycle(
            client, obs, args.hands, args.gripper_close, args.gripper_open,
            args.gripper_duration, args.pose_frame, args.send_gripper,
        )

    if not args.url:
        if dump['mapping_error'] or dump['state'] is None:
            print('[probe] local observation is not ready for inference')
            return 1
        print('[probe] local observation looks usable. Pass --url to handshake the server.')
        return 0

    if dump['mapping_error'] or dump['state'] is None:
        print('[probe] refusing to call the server until the observation dump is valid')
        return 1

    metadata, image_lines, jump_lines, jump_ok = infer_once(
        url=args.url,
        obs=obs,
        instruction=args.instruction,
        unnorm_key=args.unnorm_key,
        camera_overrides=camera_overrides,
        connect_timeout=args.connect_timeout,
        max_jump=args.max_first_step_jump,
    )
    print('[probe] server metadata:')
    print(json.dumps(metadata, indent=2, default=str))
    print('\n'.join(image_lines))
    print('\n'.join(jump_lines))
    if not jump_ok:
        return 1
    print('[probe] handshake and one inference succeeded; still no motion was sent')
    return 0


if __name__ == '__main__':
    sys.exit(main())
