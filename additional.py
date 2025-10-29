import importlib.util
import random
import sys
import time

import numpy as np

from llm_ik import BOUND, difference_angle, difference_distance, reached, Robot, SEED

# The total number of tests.
TESTS = 100

# Ensure consistent results.
random.seed(SEED)
np.random.seed(SEED)

# Import the best LLM-IK solution.
full_path = "Solutions/UR5/Google Gemini 2.5 Pro/0-5-Transform-Extend.py"
module_name = "llm_ik_solution"
spec = importlib.util.spec_from_file_location(module_name, full_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
inverse_kinematics = getattr(module, "inverse_kinematics")

# Create the various solvers.
robot_llm_ik = Robot("UR5")
links = robot_llm_ik.chains[0][5].links
bounds_llm_ik = []
for link in links:
    if link.joint_type == "fixed":
        continue
    if link.bounds is None or link.bounds == (-np.inf, np.inf):
        bounds_llm_ik.append((-BOUND, BOUND))
    else:
        bounds_llm_ik.append(link.bounds)

# Define data caches.
cache = {}
for entry in ["IKPy", "LLM-IK"]:
    cache[entry] = {"Success": 0, "Elapsed": 0, "Distance": 0, "Angle": 0}


def common_caching(elapsed: float, distance: float, angle: float, title: str) -> None:
    """
    Common solver caching operations.
    :param elapsed: The elapsed execution time.
    :param distance: The positional error.
    :param angle: The orientation error.
    :param title: Where to cache it.
    :return: If it was a success, the elapsed execution time, and the distance and angle offsets.
    """
    if reached(distance, angle):
        cache[title]["Success"] += 1
    else:
        cache[title]["Distance"] += distance
        cache[title]["Angle"] += angle
    cache[title]["Elapsed"] += elapsed


def ikpy_common(case: list[float]) -> tuple[list[float], list[float]]:
    """
    Apply clamping for the IKPy-based solvers.
    :param case: The joints to test for this case.
    :return: The target position and orientation to solve for.
    """
    adjusted = []
    for joint in range(6):
        adjusted.append(min(max(case[joint], -BOUND), BOUND))
    positions, orientations = robot_llm_ik.forward_kinematics(0, 5, adjusted)
    return positions[-1], orientations[-1]


def test_ikpy(case: list[float]) -> None:
    """
    Test built-in IKPy.
    :param case: The joints to test for this case.
    :return: Nothing.
    """
    target_position, target_orientation = ikpy_common(case)
    _, distance, angle, elapsed = robot_llm_ik.inverse_kinematics(0, 5, target_position, target_orientation)
    common_caching(elapsed, distance, angle, "IKPy")


def test_llm_ik(case: list[float]) -> None:
    """
    Test the LLM-IK solution.
    :param case: The joints to test for this case.
    :return: Nothing.
    """
    target_position, target_orientation = ikpy_common(case)
    start_time = time.perf_counter()
    solution = inverse_kinematics(target_position, target_orientation)
    elapsed = time.perf_counter() - start_time
    positions, orientations = robot_llm_ik.forward_kinematics(0, 5, solution)
    distance = difference_distance(target_position, positions[-1])
    angle = difference_angle(target_orientation, orientations[-1])
    common_caching(elapsed, distance, angle, "LLM-IK")


# Run all test cases.
for i in range(TESTS):
    # Generate random joints between -2 Pi and 2 Pi.
    joints = [random.uniform(-BOUND, BOUND) for _ in range(6)]
    test_ikpy(joints)
    test_llm_ik(joints)


# Tabulate data.
s = "Solver,Success Rate (%),Average Elapsed Time (s),Average Failure Distance,Average Failure Angle (°)"
# TODO - Order by best to worst.
for entry in cache:
    failures = TESTS - cache[entry]["Success"]
    if failures < 1:
        d = "-"
        a = "-"
    else:
        d = cache[entry]["Distance"] / failures
        a = f"{cache[entry]['Distance'] / failures} °"
    s += f"\n{entry},{cache[entry]['Success'] / TESTS * 100}%,{cache[entry]['Elapsed'] / TESTS} s,{d},{a}"
with open("Additional.csv", "w") as file:
    file.write(s)
