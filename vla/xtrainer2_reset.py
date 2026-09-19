"""Move the arms to a named start pose before the first policy command.

Each task has its own 16-D initial state, ``[left xyz+xyzw+gripper, right same]``.
Select one with ``--init-pose``. Add a task by dropping another array into
``INIT_POSES``; the flag's choices update from that table.
"""
import numpy as np

from vla.xtrainer2_contract import ACTION_DIM, LEFT_GRIPPER, RIGHT_GRIPPER, build_state, split_action

#: Vendor xtrainer ``start_pose``. Grippers open.
DEFAULT_INIT_POSE = np.array([
    -0.10880905, -0.35168106, 0.41723090, -0.00190030,
     0.99997574, -0.00161926, 0.00650238, 0.97254902,
     0.10591296, -0.35737094, 0.42194498, 0.01588704,
     0.99972773, -0.01559226, 0.00699642, 0.95686275,
], dtype=np.float64)

#: Task 2, purple eyebrow gel tube. First frame of episode_000076.
#: The dump is xyz + wxyz. This robot's TorsoEE command is xyz + xyzw, so the
#: quaternion is reordered here. Sending the dump unchanged rotates the wrist
#: about 90 degrees.
EYEBROW_INIT_POSE = np.array([
    -0.115665, -0.166341,  0.374760,
     0.038017,  0.999230, -0.009030,  0.003662,
     0.977653,
     0.059755, -0.328259,  0.469534,
    -0.043432, -0.994492,  0.073019,  0.061387,
     0.946473,
], dtype=np.float64)

#: pack_phone. First frame of episode_000071, near the dataset centre.
#: Same conversion: source quaternion is wxyz, stored here as xyzw.
PACK_PHONE_INIT_POSE = np.array([
    -0.121647, -0.303218,  0.421112,
    -0.000746,  0.999216, -0.010508,  0.038163,
     0.976638,
     0.146137, -0.346938,  0.430635,
    -0.076145, -0.996099,  0.044399,  0.004179,
     0.948569,
], dtype=np.float64)

INIT_POSES = {
    'default': DEFAULT_INIT_POSE,
    'eyebrow': EYEBROW_INIT_POSE,
    'pack_phone': PACK_PHONE_INIT_POSE,
}

#: Backward-compatible name for the default pose.
INIT_TARGET = DEFAULT_INIT_POSE


def rotation_deg(q0, q1):
    """Geodesic angle between two xyzw quaternions, in degrees."""
    q0 = np.asarray(q0, dtype=np.float64).reshape(4)
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = abs(float(np.clip(np.dot(q0, q1), -1.0, 1.0)))
    return float(np.degrees(2.0 * np.arccos(dot)))


def resolve_init_pose(name):
    """Return a copy of the named 16-D start pose."""
    try:
        pose = INIT_POSES[name]
    except KeyError:
        known = ', '.join(sorted(INIT_POSES))
        raise KeyError(f'unknown init pose {name!r}; known: {known}') from None
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose.shape != (ACTION_DIM,):
        raise ValueError(f'init pose {name!r} has shape {pose.shape}, expected ({ACTION_DIM},)')
    return pose.copy()


def state_from_observation(obs):
    """16-D xyzw state from a live ``RobotObservation``."""
    return build_state(
        left_pose=obs.get_observation(name='left_arm').multibody_pose,
        left_gripper=obs.get_observation(name='left_hand').multibody_state,
        right_pose=obs.get_observation(name='right_arm').multibody_pose,
        right_gripper=obs.get_observation(name='right_hand').multibody_state,
    )


def _slerp(q0, q1, t):
    q0 = np.asarray(q0, dtype=np.float64).reshape(4)
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    n0 = np.linalg.norm(q0)
    n1 = np.linalg.norm(q1)
    if n0 < 1e-8 or n1 < 1e-8:
        raise ValueError('quaternion is too small to interpolate')
    q0 = q0 / n0
    q1 = q1 / n1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta = np.arccos(dot)
    return (np.sin((1.0 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


def interpolate_to_target(current, target, n_steps):
    """Linear xyz/gripper and slerp xyzw from ``current`` to ``target``.

    Returns ``(n_steps, 16)`` including the target and excluding ``current``,
    matching the vendor clients' ``init_traj[1:]``.
    """
    current = np.asarray(current, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if current.shape != (ACTION_DIM,) or target.shape != (ACTION_DIM,):
        raise ValueError(
            f'expected {ACTION_DIM}-D current and target; got {current.shape} and {target.shape}')
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError(f'n_steps must be >= 1, got {n_steps}')

    left_c, left_g_c, right_c, right_g_c = split_action(current)
    left_t, left_g_t, right_t, right_g_t = split_action(target)
    traj = np.empty((n_steps, ACTION_DIM), dtype=np.float64)
    for i in range(n_steps):
        t = (i + 1) / float(n_steps)
        traj[i, 0:3] = (1.0 - t) * left_c[:3] + t * left_t[:3]
        traj[i, 3:7] = _slerp(left_c[3:], left_t[3:], t)
        traj[i, LEFT_GRIPPER] = (1.0 - t) * left_g_c + t * left_g_t
        traj[i, 8:11] = (1.0 - t) * right_c[:3] + t * right_t[:3]
        traj[i, 11:15] = _slerp(right_c[3:], right_t[3:], t)
        traj[i, RIGHT_GRIPPER] = (1.0 - t) * right_g_c + t * right_g_t
    return traj
