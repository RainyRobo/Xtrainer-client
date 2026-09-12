"""Process-local import isolation for this client.

Only touches sys.path for the current Python process. Does not change
PYTHONPATH, site-packages, or anyone else's client. Shared system SDKs (ROS,
numpy, OpenCV, ...) stay on the existing path.

Code we did not write lives under ./deps so the repo root stays readable. Those
packages are still imported by their own top-level names (``import data_msgs``,
``import data_pipeline...``), so ./deps itself goes on sys.path.

deps/vendor is a *fallback*, appended rather than prepended: it supplies packages
the machine lacks without overriding ones it already has. The wheels there are
built for Python 3.10+, while the robot container runs 3.8 -- shadowing its
working websockets/msgpack with them breaks every import.
"""
import os
import sys

CLIENT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEPS_ROOT = os.path.join(CLIENT_ROOT, "deps")
VENDOR_ROOT = os.path.join(DEPS_ROOT, "vendor")
_ROBOT_PYTHON = os.environ.get(
    "XTRAINER_ROBOT_PYTHON",
    "/data/robotics.install/lib/python3/dist-packages",
)
_LIVE_MSG_PACKAGES = (
    "data_msgs",
    "perception_msgs",
    "multibody_msgs",
    "audio_common_msgs",
)


def _place(path, first):
    if not path or not os.path.isdir(path):
        return
    try:
        sys.path.remove(path)
    except ValueError:
        pass
    if first:
        sys.path.insert(0, path)
    else:
        sys.path.append(path)


def _prepend(path):
    _place(path, first=True)


def _prefer_live_robot_msgs():
    """Use the robot's genpy messages when this machine has them.

    The copies in this repo can drift. rospy silently drops /robot/observations
    when the MD5 does not match the publisher.
    """
    if not os.path.isdir(os.path.join(_ROBOT_PYTHON, "data_msgs")):
        return
    link_root = os.path.join(CLIENT_ROOT, ".live_ros_msgs")
    try:
        os.makedirs(link_root, exist_ok=True)
    except OSError:
        return
    for package in _LIVE_MSG_PACKAGES:
        src = os.path.join(_ROBOT_PYTHON, package)
        dst = os.path.join(link_root, package)
        if not os.path.isdir(src):
            continue
        try:
            if os.path.islink(dst) or os.path.exists(dst):
                if os.path.realpath(dst) == os.path.realpath(src):
                    continue
                os.remove(dst)
            os.symlink(src, dst)
        except OSError:
            continue
    _prepend(link_root)


_place(VENDOR_ROOT, first=False)
_prepend(DEPS_ROOT)
_prepend(CLIENT_ROOT)
_prefer_live_robot_msgs()
