# Taken and adapted from https://github.com/devrimcavusoglu/pybboxes/blob/main/tests/utils.py

import json
import os
import re
import shutil
import sys
from pathlib import Path

from deepdiff import DeepDiff

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_json(path: str):
    with open(path, "r") as jf:
        content = json.load(jf)
    return content


def shell(command, exit_status=0):
    """
    Run command through shell and return exit status if exit status of command run match with given exit status.

    Args:
        command: (str) Command string which runs through system shell.
        exit_status: (int) Expected exit status of given command run.

    Returns: actual_exit_status

    """
    actual_exit_status = os.system(command)
    if actual_exit_status == exit_status:
        return 0
    return actual_exit_status


def validate_and_exit(expected_out_status=0, **kwargs):
    if all([arg == expected_out_status for arg in kwargs.values()]):
        # Expected status, OK
        sys.exit(0)
    else:
        # Failure
        print_console_centered("Summary Results")
        fail_count = 0
        for component, exit_status in kwargs.items():
            if exit_status != expected_out_status:
                print(f"{component} failed.")
                fail_count += 1
        print_console_centered(f"{len(kwargs)-fail_count} success, {fail_count} failure")
        sys.exit(1)


def print_console_centered(text: str, fill_char="="):
    w, _ = shutil.get_terminal_size((80, 20))
    print(f" {text} ".center(w, fill_char))


def assert_almost_equal(actual, desired, decimal=3, exclude_paths=None, **kwargs):
    # significant digits default value changed to 3 (from 5) due to variety in
    # results for different hardware architectures.
    diff = DeepDiff(actual, desired, significant_digits=decimal, exclude_paths=exclude_paths, **kwargs)
    assert (
        diff == {}
    ), f"Actual and Desired Dicts are not Almost Equal:\n {json.dumps(diff, indent=2, default=str)}"


def shell_capture(command, out_json=True):
    out = os.popen(command).read()
    if out_json:
        out = re.findall(r"{\s+.*\}", out, flags=re.MULTILINE | re.DOTALL)[0].replace("\n", "")
        return json.loads(out)
    return out
