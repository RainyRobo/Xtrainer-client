#!/usr/bin/env python3
"""Minimal /robot/actions driver, for learning the SDK by hand.

Run inside the robot Noetic docker::

    cd /data/arianliu/client
    python3 tools/simple_control.py state --debug
    python3 tools/simple_control.py gripper --left 0.0 --debug --dry-run
    python3 tools/simple_control.py gripper --left 0.0 --seconds 2 --debug
    python3 tools/simple_control.py move --arm left --dz 0.02 --yes

The vendor's own inference service streams the same command at 100 Hz instead
of publishing once, so every subcommand here streams too.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401

import argparse
import time

import numpy as np

MAX_DELTA = 0.05


def connect(wait_timeout=15.0):
    import rosgraph
    import rospy
    from data_pipeline.tools.robot_ros_client import RobotRosClient

    if not rosgraph.is_master_online():
        raise RuntimeError('ROS master is not running; start the robot stack first')

    rospy.init_node('xtrainer2_simple_control', anonymous=True)
    client = RobotRosClient(wait_first_msg=False, frame_id='xtrainer2_simple_control')
    deadline = time.time() + wait_timeout
    while time.time() < deadline and not rospy.is_shutdown():
        if client.obs_msg is not None:
            obs = client.get_observation(display_image=False)
            if obs is not None:
                return client, obs
        rospy.sleep(0.2)
    raise RuntimeError(f'no /robot/observations within {wait_timeout:.0f}s')


def read_pose(obs, name):
    component = obs.get_observation(name=name)
    if component is None or component.multibody_pose is None:
        raise RuntimeError(f'{name} has no multibody_pose')
    return np.asarray(component.multibody_pose, dtype=np.float64).reshape(7)


def read_gripper(obs, name):
    component = obs.get_observation(name=name)
    if component is None or component.multibody_state is None:
        raise RuntimeError(f'{name} has no multibody_state')
    return float(np.asarray(component.multibody_state, dtype=np.float64).reshape(-1)[0])


def print_state(obs, prefix=''):
    for side in ('left', 'right'):
        pose = read_pose(obs, f'{side}_arm')
        grip = read_gripper(obs, f'{side}_hand')
        xyz = ', '.join(f'{v:+.4f}' for v in pose[:3])
        quat = ', '.join(f'{v:+.4f}' for v in pose[3:])
        print(f'{prefix}{side:>5}  xyz=[{xyz}]  xyzw=[{quat}]  gripper={grip:.3f}')


def raw_gripper(client, name):
    """Read a gripper straight off the cached message, skipping image decode."""
    for comp in client.obs_msg.component_observations:
        if comp.header.frame_id == name and len(comp.multibody_state.states) > 0:
            return comp.multibody_state.states[0].q
    return None


def print_topology():
    import rosgraph

    master = rosgraph.Master('/xtrainer2_simple_control')
    pubs, subs, _ = master.getSystemState()

    def nodes_for(table, topic):
        for name, nodes in table:
            if name == topic:
                return sorted(nodes)
        return []

    for topic in ('/robot/observations', '/robot/actions'):
        p = nodes_for(pubs, topic) or ['none']
        s = nodes_for(subs, topic) or ['none']
        print(f'  {topic:<22} pub={",".join(p):<28} sub={",".join(s)}')


def robot_layout(client):
    """The component order and joint count the robot reports for itself."""
    return [(c.header.frame_id,
             c.multibody_pose.header.frame_id,
             len(c.multibody_state.states))
            for c in client.obs_msg.component_observations]


def print_layout(layout):
    for i, (name, frame, dof) in enumerate(layout):
        print(f'  [{i}] {name:<12} pose_frame={frame!r:<10} dof={dof}')


def inspect_message(msg, layout):
    """Print the outgoing message and return a list of suspected problems."""
    print('outgoing component_actions:')
    problems = []
    for i, comp in enumerate(msg.component_actions):
        o = comp.pose_command.pose.orientation
        p = comp.pose_command.pose.position
        joints = list(comp.joint_commands)
        print(f'  [{i}] {comp.header.frame_id:<12} '
              f'frame={comp.pose_command.header.frame_id!r:<10} '
              f'xyz=({p.x:+.4f},{p.y:+.4f},{p.z:+.4f}) '
              f'xyzw=({o.x:+.4f},{o.y:+.4f},{o.z:+.4f},{o.w:+.4f}) '
              f'joints={joints} duration={comp.duration}')
        if not any((o.x, o.y, o.z, o.w)):
            problems.append(f'[{i}] {comp.header.frame_id}: all-zero quaternion, '
                            'xtrainer_main throws std::logic_error on this')

    sent_names = [c.header.frame_id for c in msg.component_actions]
    robot_names = [name for name, _, _ in layout]
    if sent_names != robot_names:
        problems.append(f'component order {sent_names} != robot order {robot_names}; '
                        'xtrainer_main indexes component_actions[i] positionally')

    dof_by_name = {name: dof for name, _, dof in layout}
    for comp in msg.component_actions:
        n = len(comp.joint_commands)
        dof = dof_by_name.get(comp.header.frame_id)
        if n and dof is not None and n != dof:
            problems.append(f'{comp.header.frame_id}: {n} joint_commands but robot reports dof={dof}')
    return problems


def build_action(poses, grippers, frame, duration):
    """poses/grippers are dicts keyed by 'left'/'right'; None means "omit".

    xtrainer_main runs QuaternionToRotationMatrix() on the pose_command of
    every component it receives. A component built without a pose carries a
    default-initialised (0, 0, 0, 0) quaternion, which makes it throw
    std::logic_error and abort, taking the whole required-node stack with it.
    So a gripper command must still carry a real pose; the arm's measured pose
    is used, which is a no-op if the robot acts on it.
    """
    import rospy
    from data_pipeline.tools.robot_action import ComponentAction, RobotAction

    actions = RobotAction()
    for side in ('left', 'right'):
        pose = poses.get(side)
        if pose is not None:
            arm = ComponentAction()
            arm.name = f'{side}_arm'
            arm.pose_command = ComponentAction.Pose(np.asarray(pose, dtype=np.float64), frame)
            arm.joint_commands = None
            arm.duration = duration
            actions.actions.append(arm)

        opening = grippers.get(side)
        if opening is not None:
            if pose is None:
                raise RuntimeError(f'refusing to send {side}_hand without a pose')
            hand = ComponentAction()
            hand.name = f'{side}_hand'
            hand.pose_command = ComponentAction.Pose(np.asarray(pose, dtype=np.float64), frame)
            hand.joint_commands = np.array([float(np.clip(opening, 0.0, 1.0))])
            hand.duration = duration
            actions.actions.append(hand)

    actions.timestamp = rospy.Time.now().to_sec()
    return actions


def stream(client, poses, grippers, frame, duration, rate_hz, seconds, debug=False):
    import rospy

    rate = rospy.Rate(rate_hz)
    sample_every = max(1, int(rate_hz / 10))
    deadline = time.time() + seconds
    sent = 0
    track = {'left_hand': [], 'right_hand': []}
    while time.time() < deadline and not rospy.is_shutdown():
        # A vanished subscriber means xtrainer_main died; knowing which message
        # it died on separates "rejects the first one" from "drifts then dies".
        if client.action_pub.get_num_connections() == 0:
            print(f'!! /robot/actions lost its subscriber after {sent} messages '
                  '-- xtrainer_main most likely just died')
            break
        client.send_action(build_action(poses, grippers, frame, duration))
        sent += 1
        if debug and sent % sample_every == 0:
            for name in track:
                value = raw_gripper(client, name)
                if value is not None:
                    track[name].append(value)
        rate.sleep()

    print(f'sent {sent} messages over {seconds:.1f}s at {rate_hz} Hz')
    if debug:
        for name, values in track.items():
            if not values:
                continue
            print(f'  {name} during stream: first={values[0]:.3f} '
                  f'min={min(values):.3f} max={max(values):.3f} last={values[-1]:.3f}'
                  f'{"  (never changed)" if max(values) - min(values) < 1e-4 else ""}')
    return sent


def wait_for_subscriber(client, timeout=3.0):
    import rospy

    deadline = time.time() + timeout
    while client.action_pub.get_num_connections() == 0 and time.time() < deadline:
        rospy.sleep(0.1)
    n = client.action_pub.get_num_connections()
    if n == 0:
        raise RuntimeError('nobody subscribes to /robot/actions')
    print(f'/robot/actions subscribers: {n}')


def preflight(args, client, poses, grippers):
    """Compare the message we are about to send against the live robot."""
    from data_pipeline.tools.robot_message_conversion import ActionConversion

    if not args.debug:
        return True

    print('ros graph:')
    print_topology()

    layout = robot_layout(client)
    print('component layout reported by /robot/observations:')
    print_layout(layout)

    action = build_action(poses, grippers, args.frame, args.duration)
    msg = ActionConversion.to_robot_message(action, 'xtrainer2_simple_control')
    problems = inspect_message(msg, layout)

    if not problems:
        print('preflight: no mismatch found')
        return True
    print('preflight problems:')
    for problem in problems:
        print(f'  ! {problem}')
    if not args.force:
        print('refusing to publish; pass --force to send anyway')
        return False
    print('--force given, publishing regardless')
    return True


def cmd_state(args, client, obs):
    print_state(obs)
    if args.debug:
        print('ros graph:')
        print_topology()
        print('component layout reported by /robot/observations:')
        print_layout(robot_layout(client))
    return 0


def cmd_gripper(args, client, obs):
    print_state(obs, prefix='before  ')

    grippers = {'left': args.left, 'right': args.right}
    sides = ('left', 'right') if args.hold_arms else \
        tuple(s for s in ('left', 'right') if grippers[s] is not None)
    poses = {side: read_pose(obs, f'{side}_arm') for side in sides}

    if not preflight(args, client, poses, grippers):
        return 1
    if args.dry_run:
        print('dry run, nothing published')
        return 0

    wait_for_subscriber(client)
    stream(client, poses, grippers, args.frame, args.duration, args.rate,
           args.seconds, debug=args.debug)

    import rospy
    rospy.sleep(0.5)
    after = client.get_observation(display_image=False) or obs
    print_state(after, prefix='after   ')
    return 0


def cmd_move(args, client, obs):
    if not args.yes:
        print('refusing to move without --yes')
        return 1
    delta = np.array([args.dx, args.dy, args.dz], dtype=np.float64)
    if np.abs(delta).max() > MAX_DELTA:
        print(f'delta {delta} exceeds {MAX_DELTA} m per axis')
        return 1

    print_state(obs, prefix='before  ')

    poses = {}
    for side in ('left', 'right'):
        pose = read_pose(obs, f'{side}_arm')
        if side == args.arm or args.arm == 'both':
            pose = pose.copy()
            pose[:3] += delta
        poses[side] = pose
    target = poses[args.arm if args.arm != 'both' else 'left'][:3]
    print(f'target xyz for {args.arm}: [' + ', '.join(f'{v:+.4f}' for v in target) + ']')

    if not preflight(args, client, poses, {}):
        return 1
    if args.dry_run:
        print('dry run, nothing published')
        return 0

    wait_for_subscriber(client)
    stream(client, poses, {}, args.frame, args.duration, args.rate,
           args.seconds, debug=args.debug)

    import rospy
    rospy.sleep(0.5)
    after = client.get_observation(display_image=False) or obs
    print_state(after, prefix='after   ')
    return 0


def build_argparser():
    # Shared options are attached to every subparser so they work on either
    # side of the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    # TorsoEE is the only frame literal xtrainer_main carries, and it is what
    # /robot/observations reports. Anything else is an unknown frame id.
    common.add_argument('--frame', default='TorsoEE', choices=['TorsoEE', ''],
                        help="pose frame; '' is what the vendor runner sends")
    common.add_argument('--rate', type=float, default=100.0,
                        help='publish rate; the vendor runner uses 100 Hz for pose')
    common.add_argument('--seconds', type=float, default=2.0,
                        help='how long to keep streaming the command')
    common.add_argument('--duration', type=float, default=0.0,
                        help='per-command duration field')
    common.add_argument('--debug', action='store_true',
                        help='dump the ros graph, the robot layout and the outgoing message')
    common.add_argument('--dry-run', action='store_true',
                        help='build and inspect the message but publish nothing')
    common.add_argument('--force', action='store_true',
                        help='publish even when the preflight finds a mismatch')

    parser = argparse.ArgumentParser(
        description='minimal xtrainer2 /robot/actions driver', parents=[common])
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('state', parents=[common],
                   help='print poses and grippers')

    gripper = sub.add_parser('gripper', parents=[common],
                             help='command one or both grippers (0 closed, 1 open)')
    gripper.add_argument('--left', type=float, default=None)
    gripper.add_argument('--right', type=float, default=None)
    gripper.add_argument('--hold-arms', action='store_true',
                         help='also hold the arm that has no gripper command')

    move = sub.add_parser('move', parents=[common],
                          help='small end-effector translation')
    move.add_argument('--arm', default='left', choices=['left', 'right', 'both'])
    move.add_argument('--dx', type=float, default=0.0)
    move.add_argument('--dy', type=float, default=0.0)
    move.add_argument('--dz', type=float, default=0.0)
    move.add_argument('--yes', action='store_true', help='required; the arm will move')
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    if args.command == 'gripper' and args.left is None and args.right is None:
        print('give --left and/or --right')
        return 1

    client, obs = connect()
    handler = {
        'state': cmd_state,
        'gripper': cmd_gripper,
        'move': cmd_move,
    }[args.command]
    return handler(args, client, obs)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
