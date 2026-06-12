"""Course plumbing: load student modules by number.

Module folders contain dashes (modules/03-a-place-to-stand), so they
can't be imported the normal way. `load(3, "experience_map")` loads the
student's file from the right folder.

Later modules import earlier ones through this loader — the dependency
chain is real: your Module 4 demo runs YOUR Module 3 agent.

(Setting the environment variable R2R_SOLUTIONS=1 loads reference
solutions instead. That switch exists for grading and for the course's
own CI. Flipping it to dodge a wall you've hit will quietly steal the
entire point of the course from you.)
"""

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODULE_DIRS = {
    1: "01-the-coin",
    2: "02-the-greedy-trap",
    3: "03-a-place-to-stand",
    4: "04-messages-from-the-future",
    5: "05-the-wall",
    6: "06-the-gradient-machine",
    7: "07-the-brain",
    8: "08-the-mirror",
    9: "09-imagination",
    10: "10-reason",
}


def load(number, name):
    """Load `name`.py from module `number`'s folder and return the module."""
    if os.environ.get("R2R_SOLUTIONS") == "1":
        path = ROOT / "solutions" / f"module{number:02d}" / f"{name}.py"
    else:
        path = ROOT / "modules" / MODULE_DIRS[number] / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"no such course file: {path}")
    spec = importlib.util.spec_from_file_location(f"r2r_m{number:02d}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
