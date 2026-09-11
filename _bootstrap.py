"""Process-local import isolation for this client.

Only prepends this directory and ./vendor for the current Python process.
Does not change PYTHONPATH, site-packages, or anyone else's client.
Shared system SDKs (ROS, numpy, OpenCV, ...) stay on the existing path.
"""
import os
import sys

CLIENT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENDOR_ROOT = os.path.join(CLIENT_ROOT, "vendor")


def _prepend(path):
    if not path or not os.path.isdir(path):
        return
    try:
        sys.path.remove(path)
    except ValueError:
        pass
    sys.path.insert(0, path)


_prepend(VENDOR_ROOT)
_prepend(CLIENT_ROOT)
