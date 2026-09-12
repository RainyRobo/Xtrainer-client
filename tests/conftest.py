"""Import the repo the same way the executables do.

pytest loads this before any test module, so the test files themselves carry no
sys.path boilerplate and cannot drift from what runs on the robot.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _bootstrap  # noqa: F401,E402
