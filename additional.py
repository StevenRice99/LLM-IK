import importlib.util
import random
import sys
import time

import numpy as np

from llm_ik import BOUND, difference_angle, difference_distance, reached, Robot, SEED
from trac_ik import TracIK
from ur_ikfast import ur_kinematics

# The total number of tests.
TESTS = 10000

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
robot_ikfast = ur_kinematics.URKinematics('ur5')
robot_trac_ik = TracIK(urdf_path="Robots/UR5.urdf", base_link_name="base_link", tip_link_name="tool0")

# Cache joint limits, just in case the representations of the robot differ slightly in different implementations.
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
for entry in ["IKPy", "LLM-IK", "IKFast", "Trac-IK"]:
    cache[entry] = {"Success": 0, "Elapsed": 0, "Distance": 0, "Angle": 0}


def get_quaternion_angle_difference(q1, q2):
    """
    Calculates the angular difference (in degrees) between two unit quaternions.
    Assumes q1 and q2 are NumPy arrays of shape (4,) or (N, 4) for batch processing.
    :param q1: The first quaternion.
    :param q2: The second quaternion.
    :return: The angle difference.
    """
    # Calculate the dot product.
    dot_product = np.dot(q1, q2)
    # Handle batch processing (N, 4) dot (N, 4).
    if q1.ndim == 2:
        dot_product = np.sum(q1 * q2, axis=1)
    # Take the absolute value to get the shortest path.
    dot_product = np.abs(dot_product)
    # Clip the value to be in [-1.0, 1.0] to prevent acos from failing due to floating point inaccuracies.
    dot_product = np.clip(dot_product, -1.0, 1.0)
    # Calculate the angle in radians which is the half-angle.
    half_angle = np.arccos(dot_product)
    # Double it to get the full angle.
    angle_rad = 2 * half_angle
    # Convert to degrees.
    return np.degrees(angle_rad)


def quaternion_from_matrix(matrix):
    """
    Get a quaternion from a matrix.
    :param matrix: The matrix.
    :return: The quaternion.
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # Symmetric matrix K.
    K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                  [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                  [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                  [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
    K /= 3.0
    # Quaternion is eigenvector of K that corresponds to the largest eigenvalue.
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


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
    positions, orientations = robot_llm_ik.forward_kinematics(0, 5, case)
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


def test_ikfast(case: list[float]) -> None:
    """
    Test the IKFast solver.
    :param case: The joints to test for this case.
    :return: Nothing.
    """
    t = robot_ikfast.forward(case)
    zeros = [0] * len(case)
    start_time = time.perf_counter()
    solution = robot_ikfast.inverse(t, False, zeros)
    elapsed = time.perf_counter() - start_time
    if solution is None:
        r = robot_ikfast.forward(zeros)
    else:
        r = robot_ikfast.forward(solution)
    distance = difference_distance([t[0], t[1], t[2]], [r[0], r[1], r[2]])
    angle = get_quaternion_angle_difference(np.array([t[3], t[4], t[5], t[6]], dtype="float64"),
                                            np.array([r[3], r[4], r[5], r[6]], dtype="float64"))
    common_caching(elapsed, distance, angle, "IKFast")


def test_trac_ik(case: list[float]) -> None:
    """
    Test the Trac-IK solver.
    :param case: The joints to test for this case.
    :return: Nothing.
    """
    lb, ub = robot_trac_ik.joint_limits
    clamped = []
    for index in range(len(case)):
        clamped.append(min(ub[index], max(case[index], lb[index])))
    t_p, t_r = robot_trac_ik.fk(np.array(clamped))
    zeros = [0] * len(case)
    zeros = np.array(zeros)
    start_time = time.perf_counter()
    solution = robot_trac_ik.ik(t_p, t_r, zeros)
    elapsed = time.perf_counter() - start_time
    if solution is None:
        r_p, r_r = robot_trac_ik.fk(zeros)
    else:
        r_p, r_r = robot_trac_ik.fk(solution)
    distance = difference_distance(t_p, r_p)
    angle = get_quaternion_angle_difference(quaternion_from_matrix(t_r), quaternion_from_matrix(r_r))
    common_caching(elapsed, distance, angle, "Trac-IK")


# Run all test cases.
for i in range(TESTS):
    # Generate random joints between the IKPy URDF limits.
    # Other solvers further clamp as needed if their limits differ.
    joints = []
    for j in range(6):
        joints.append(random.uniform(bounds_llm_ik[j][0], bounds_llm_ik[j][1]))
    # Test each model.
    test_ikpy(joints)
    test_llm_ik(joints)
    test_ikfast(joints)
    test_trac_ik(joints)


# Tabulate data.
s = "Solver,Success Rate (%),Average Elapsed Time (s),Average Failure Distance,Average Failure Angle (°)"
cache = dict(sorted(cache.items(), key=lambda item: (
    -item[1]["Success"],
    item[1]["Elapsed"],
    item[0]
)))
for entry in cache:
    failures = TESTS - cache[entry]["Success"]
    if failures < 1:
        d = "-"
        a = "-"
    else:
        d = cache[entry]["Distance"] / failures
        a = f"{cache[entry]['Distance'] / failures} °"
    s += f"\n{entry},{cache[entry]['Success'] / TESTS * 100}%,{cache[entry]['Elapsed'] / TESTS} s,{d},{a}"
with open("Additional.csv", "w", encoding="utf-8", errors="ignore") as file:
    file.write(s)
