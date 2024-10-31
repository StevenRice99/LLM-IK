import math
import random
import importlib
import importlib.util
import time
import traceback
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
import dm_control.utils.inverse_kinematics
import os
from collections.abc import Iterable
from dm_control import mujoco as mujoco_dm
from dm_control.mujoco.wrapper import mjbindings
from tabulate import tabulate


def import_model(name: str) -> [mujoco.MjModel, str]:
    """
    Import a file into Mujoco.
    :param name: The name of the robot to load.
    :return: The Mujoco model and the path to the model.
    """
    path = os.path.join(os.getcwd(), "Models", name, "Model.xml")
    if not os.path.exists(path):
        print(f"Model {name} not found at {path}.")
        return None, path
    # The method is meant to be used like this, but the Mujoco API itself is defined wrong.
    # This is to hide that warning which we are otherwise handling properly.
    # noinspection PyArgumentList
    model = mujoco.MjModel.from_xml_path(filename=path, assets=None)
    return model, path


def quaternion_to_rotation_matrix(quat) -> np.ndarray[Any, np.dtype[np.floating | np.float_]]:
    """
    Convert a quaternion into a rotation matrix.
    :param quat: The quaternion as [x, y, z, w].
    :return: The rotation matrix.
    """
    x, y, z, w = quat
    # Compute the rotation matrix elements.
    r = np.zeros((3, 3))
    r[0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    r[0, 1] = 2 * (x * y - z * w)
    r[0, 2] = 2 * (x * z + y * w)
    r[1, 0] = 2 * (x * y + z * w)
    r[1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    r[1, 2] = 2 * (y * z - x * w)
    r[2, 0] = 2 * (x * z - y * w)
    r[2, 1] = 2 * (y * z + x * w)
    r[2, 2] = 1 - 2 * (x ** 2 + y ** 2)
    return r


def dh_table(dh_parameters: list) -> str:
    """
    Prints the DH parameters in a human-readable table.
    :param dh_parameters: A list of DH parameters for each joint.
    :return: The table as a string.
    """
    headers = ["Joint", "Theta (rad)", "d (m)", "a (m)", "Alpha (rad)"]
    table = []
    for i, params in enumerate(dh_parameters):
        theta_i, d_i, a_i, alpha_i = params
        table.append([i + 1, neat(theta_i), neat(d_i), neat(a_i), neat(alpha_i)])
    return tabulate(table, headers=headers, tablefmt="github")


def get_data(model: mujoco.MjModel, limits: bool = False,
             collisions: bool = False, tcp: int = 0) -> [mujoco.MjData, list, list, list, list, list, list, str, list,
                                                         int]:
    """
    Get the data for the Mujoco model.
    :param model: The Mujoco model.
    :param limits: If limits should be enforced or not.
    :param collisions: If collisions should be enforced or not.
    :param tcp: The index of the TCP we want.
    :return: The data of the Mujoco model, joint positions, joint orientations, joint axes, joint lower bounds, joint
    upper bounds, the types of joints, the name of the site for the end effector, the names of the joints, and the TCP index.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    positions = []
    orientations = []
    axes = []
    lower = []
    upper = []
    types = []
    joints = []
    # No previous offset for the first joint.
    previous_offset = np.array([0, 0, 0])
    # How many
    if tcp < 1 or tcp > model.nsite:
        tcp = model.nsite
    tcp_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, tcp - 1)
    tcp_id = model.site_bodyid[tcp - 1]
    keep = []
    tcp_id = model.body_parentid[tcp_id]
    while tcp_id != 0:
        keep.append(tcp_id)
        tcp_id = model.body_parentid[tcp_id]
    # Loop every joint.
    for i in range(model.njnt):
        joints.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))
        # Get the type of joint.
        joint_type = model.jnt_type[i]
        if joint_type == 3:
            joint_type = "Revolute"
        elif joint_type == 2:
            joint_type = "Linear"
        elif joint_type == 1:
            joint_type = "Spherical"
        else:
            joint_type = "Free"
        types.append(joint_type)
        body = model.jnt_bodyid[i]
        # Not interested in later joints
        if body not in keep:
            continue
        # Some joints may be offset from their body, so we need to account for that.
        offset = model.jnt_pos[i]
        # Get this position taking into account how much the previous joint was offset.
        positions.append( model.body_pos[body] + offset - previous_offset)
        # Set the offset for the next iteration.
        previous_offset = offset
        orientations.append(model.body_quat[body])
        axes.append(model.jnt_axis[i])
        # Get limits.
        if model.jnt_limited[i]:
            # If we are not looking to enforce limits, remove them.
            if not limits:
                model.jnt_limited[i] = 0
                model.jnt_range[i][0] = -3.14159
                model.jnt_range[i][1] = 3.14159
            lower.append(model.jnt_range[i][0])
            upper.append(model.jnt_range[i][1])
        else:
            lower.append(-3.14159)
            upper.append(3.14159)
    # Turn off collisions if needed.
    if not collisions:
        for i in range(model.ngeom):
            model.geom_contype[i] = 0
            model.geom_conaffinity[i] = 0
            model.geom_condim[i] = 1
    # If there are collisions, turn them off for joints we do not care about.
    else:
        for i in range(model.ngeom):
            body = model.geom_bodyid[i]
            if body in keep:
                continue
            model.geom_contype[i] = 0
            model.geom_conaffinity[i] = 0
            model.geom_condim[i] = 1
    return data, positions, orientations, axes, lower, upper, types, tcp_name, joints, tcp,


def neat(value: float) -> str:
    """
    Format a float value with no trailing zeros.
    :param value: The float value.
    :return: The value as a formatted string.
    """
    return f"{value:.8f}".rstrip('0').rstrip('.')


def generate_prompt(name: str, orientation: bool = False, limits: bool = False, tcp: int = 0,
                    iterative: str or None = None) -> str:
    """
    Generate the prompt to give to LLMs.
    :param name: The name of the robot to generate the prompt for.
    :param orientation: If orientation should be solved for.
    :param limits: If limits should be enforced or not.
    :param tcp: The index of the TCP we want.
    :param iterative: The solver so that this can look at a lower-joints solver to try and help it.
    :return: The prompt for the LLM.
    """
    model, path = import_model(name)
    if path is None:
        return ""
    data, positions, orientations, axes, lower, upper, types, site, joints, tcp = get_data(model, limits, False, tcp)
    # Load an example from a lower-level joint to help the solver.
    example = None
    if iterative is not None:
        path = os.path.join(os.getcwd(), "Solvers", name, "Orientation" if orientation else "Position", str(tcp - 1),
                            f"{iterative}.py")
        if os.path.exists(path):
            with open(path, "r") as file:
                example = file.read().strip()
    # Get the start of the prompt.
    details = ('<SUMMARY>\n'
               f'Produce a closed-form analytical solution for the inverse kinematics of the {len(positions)} '
               'degree-of-freedom serial manipulator detailed in the "DETAILS" section by completing the Python '
               'function provided in the "CODE" section. Positions and orientations are local coordinates relative to '
               'their parent link.')
    if limits:
        details += " Limits are in radians."
    details += (" A right-handed coordinate system is used. The X-axis is forwards, Y-axis is left, and Z-axis is up. "
                "You may assume all requested targets to solve the inverse kinematics for will be valid.")
    if example is not None:
        details += (f" A solution to solving the sub-chain of the first {len(positions) - 1} joints is provided in the "
                    '"EXISTING" section. This solution was obtained with the same serial manipulator, except "Joint '
                    f'{len(positions)}" was the "End Effector" instead of a joint. This means it had the same position'
                    f' and orientation as "Joint {len(positions)}" does now, but was the "End Effector" for that '
                    "sub-chain.")
    # Write the base nicely.
    pos = data.xpos[1]
    pos = f"[{neat(pos[0])}, {neat(pos[1])}, {neat(pos[2])}]"
    details += f"\n</SUMMARY>\n<DETAILS>\nBase = Position: {pos}"
    quat = data.xquat[1]
    quat = f"[{neat(quat[0])}, {neat(quat[1])}, {neat(quat[2])}, {neat(quat[3])}]"
    details += f", Orientation: {quat}"
    # Write all joints nicely.
    n_joints = len(positions)
    for i in range(n_joints):
        details += (f"\nJoint {i + 1}" if n_joints > 1 else "\nJoint")
        details += f" = Type: {types[i]}"
        pos = positions[i]
        pos = f"[{neat(pos[0])}, {neat(pos[1])}, {neat(pos[2])}]"
        details += f", Position: {pos}"
        quat = orientations[i]
        quat = f"[{neat(quat[0])}, {neat(quat[1])}, {neat(quat[2])}, {neat(quat[3])}]"
        details += f", Orientation: {quat}"
        axis = axes[i]
        axis = f"[{neat(axis[0])}, {neat(axis[1])}, {neat(axis[2])}]"
        details += f", Axes: {axis}"
        if limits and model.jnt_limited[i]:
            details += f", Lower: {neat(lower[i])}, Upper: {neat(upper[i])}"
    # Write the end effector nicely.
    site_id = model.nsite - 1
    body = model.site_bodyid[site_id]
    pos = model.body_pos[body]
    pos = pos + model.site_pos[site_id]
    pos = f"[{neat(pos[0])}, {neat(pos[1])}, {neat(pos[2])}]"
    quat = model.site_quat[site_id]
    quat = f"[{neat(quat[0])}, {neat(quat[1])}, {neat(quat[2])}, {neat(quat[3])}]"
    # Write the remainder of the prompt.
    details += (f"\nEnd Effector = Position: {pos}, Orientation: {quat}\n"
                "</DETAILS>\n"
                "<CODE>\n"
                "def inverse_kinematics(p: list")
    if orientation:
        details += ", q: list"
    details += (") -> list[float]:\n"
                '    """\n'
                "    Solve the inverse kinematics to reach the position"
                f"{' and orientation, both' if orientation else ''} in the reference frame of the chain's origin.\n"
                "    :param p: The position for the end effector to reach in a Cartesian [x, y, z].\n")
    if orientation:
        details += ("    :param q: The orientation for the end effector to reach in radians in a quaternion [w, x, y, z"
                    "].\n")
    details += "    :return: A list of the joint values in radians needed for the end effector to reach the position"
    if orientation:
        details += " and orientation"
    details += (".\n"
                '    """\n'
                "</CODE>")
    if example is not None:
        details += f"\n<EXISTING>\n{example}\n</EXISTING>"
    return details


def load_model(name: str, orientation: bool = False, limits: bool = False, collisions: bool = False,
               tcp: int = 0) -> [mujoco.MjModel, mujoco.MjData, list, list, str, str, dict, list, int]:
    """
    Load a model into Mujoco.
    :param name: The name of the file to load.
    :param orientation: If orientation should be solved for.
    :param limits: If limits should be enforced or not.
    :param collisions: If collisions should be enforced or not.
    :param tcp: The index of the TCP we want.
    :return: The Mujoco model, the data of the Mujoco model, joint lower bounds, joint upper bounds, the name of the
    site for the end effector, the path to the model, all LLM solvers, the names of the joints, and the index of the
    TCP.
    """
    # Do regular Mujoco set up and get joints and other data we need.
    model, path = import_model(name)
    if model is None:
        return [None, None, None, None, None, None, path, None]
    data, positions, orientations, axes, lower, upper, types, site, joints, tcp = get_data(model, limits, collisions,
                                                                                           tcp)
    # Load in solvers which exist for this robot.
    solvers = []
    folder = "Transform" if orientation else "Position"
    solvers_directory = os.path.join(os.getcwd(), "Solvers", name, folder, str(tcp))
    if os.path.exists(solvers_directory):
        for file in os.listdir(solvers_directory):
            # Ensure we are only checking Python files.
            if not file.endswith(".py"):
                continue
            # Keep the name of the file and bind the inverse kinematics method.
            # noinspection PyBroadException
            try:
                module_name = file[:-3]
                module_path = os.path.join(solvers_directory, file)
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                # Ensure it has the proper module.
                if not hasattr(module, "inverse_kinematics"):
                    continue
                method = getattr(module, "inverse_kinematics")
                solvers.append({"Name": module_name, "Method": method})
            except Exception:
                continue
    return model, data, lower, upper, site, path, solvers, joints, tcp


def set_joints(model: mujoco.MjModel, data: mujoco.MjData, values: list) -> None:
    """
    Set the Mujoco model to the specified joint values.
    :param model: The Mujoco model.
    :param data: The data for the Mujoco model.
    :param values: The joint values to set.
    :return: Nothing.
    """
    if isinstance(values, Iterable):
        number = min(model.njnt, len(values))
        for i in range(number):
            current = data.qpos[i]
            # noinspection PyBroadException
            try:
                data.ctrl[i] = values[i]
                data.qpos[i] = values[i]
            except Exception:
                data.ctrl[i] = current
                data.qpos[i] = current
            data.qvel[i] = 0
    # noinspection PyBroadException
    try:
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
    except Exception:
        return


def random_positions(model: mujoco.MjModel, data: mujoco.MjData, lower: list, upper: list) -> None:
    """
    Set the Mujoco model to random joint values.
    :param model: The Mujoco model.
    :param data: The data for the Mujoco model.
    :param lower: The joint lower bounds.
    :param upper: The joint upper bounds.
    :return: Nothing.
    """
    values = []
    for i in range(model.njnt):
        values.append(random.uniform(lower[i], upper[i]))
    set_joints(model, data, values)


def mid_positions(model: mujoco.MjModel, data: mujoco.MjData, lower: list, upper: list) -> None:
    """
    Set the Mujoco model to middle joint values.
    :param model: The Mujoco model.
    :param data: The data for the Mujoco model.
    :param lower: The joint lower bounds.
    :param upper: The joint upper bounds.
    :return: Nothing.
    """
    values = []
    for i in range(model.njnt):
        values.append((lower[i] + upper[i]) / 2)
    set_joints(model, data, values)


def get_joints(model: mujoco.MjModel, data: mujoco.MjData, joints: int = 0) -> list:
    """
    Get the joint values of the Mujoco model.
    :param model: The Mujoco model.
    :param data: The data for the Mujoco model.
    :param joints: The number of joints we care about the values for.
    :return: The joint values of the Mujoco model.
    """
    if joints < 1 or joints > model.njnt:
        joints = model.njnt
    values = []
    for i in range(joints):
        values.append(data.qpos[i])
    return values


def get_pose(model: mujoco.MjModel, data: mujoco.MjData, tcp: int = 0) -> [list, list]:
    """
    Get the current pose of the position and orientation of the end effector of the Mujoco model.
    :param model: The Mujoco model.
    :param data: The data for the Mujoco model.
    :param tcp: The index of the TCP we want.
    :return: The position [X, Y, Z] and the orientation [W, X, Y, Z].
    """
    if tcp < 1 or tcp > model.nsite:
        tcp = model.nsite
    pos = data.site_xpos[tcp - 1]
    raw = data.site_xmat[tcp - 1]
    quat = np.empty_like(data.xquat[model.site_bodyid[tcp - 1]])
    mjbindings.mjlib.mju_mat2Quat(quat, raw)
    quat /= quat.ptp()
    return [pos[0], pos[1], pos[2]], [quat[0], quat[1], quat[2], quat[3]]


def quaternion_to_euler(quat: list) -> [float, float, float]:
    """
    Convert a quaternion to Euler angles (ZYX convention).
    :param quat: The quaternion to convert.
    :return: The euler angles [X, Y, Z].
    """
    w, x, y, z = quat
    # Roll (x-axis rotation).
    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)
    sin_pitch = 2.0 * (w * y - z * x)
    # Pitch (y-axis rotation).
    # Use 90 degrees if out of range.
    pitch = math.copysign(math.pi / 2, sin_pitch) if abs(sin_pitch) >= 1 else math.asin(sin_pitch)
    # Yaw (z-axis rotation).
    sin_yaw_cos_pitch = 2.0 * (w * z + x * y)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)
    return roll, pitch, yaw


def deepmind_ik(model: mujoco.MjModel, data: mujoco.MjData, path: str, site: str, pos: list,
                quat: list or None = None, joints: None or list = None, max_steps: int = 1000) -> float:
    """
    Run the inverse kinematics from DeepMind Controls.
    :param model: The Mujoco model.
    :param data: The data for the Mujoco model.
    :param path: The path for the Mujoco model to reload the model in the DeepMind Controls instance.
    :param site: The name of the site or end effector.
    :param pos: The position for the end effector to reach.
    :param quat: The orientation for the end effector to reach.
    :param joints: The joints that can be used to solve.
    :param max_steps: How many iterations the DeepMind IK can run for.
    :return: The time to took to complete the inverse kinematics.
    """
    physics = mujoco_dm.Physics.from_xml_path(path)
    # Copy positions as it seems they might get modified during use.
    pos_copy = pos.copy()
    quat_copy = None if quat is None else quat.copy()
    start_time = time.time()
    values = dm_control.utils.inverse_kinematics.qpos_from_site_pose(physics, site, pos_copy, quat_copy, joints,
                                                                     max_steps=max_steps)
    end_time = time.time()
    set_joints(model, data, values.qpos)
    return end_time - start_time


def eval_ik(title: str, pos: list, goal_pos: list, quat: list or None = None, goal_quat: list or None = None,
            duration: float = 0, error: float = 0.001, joints: list or None = None, solution: list or None = None,
            verbose: bool = False) -> [bool, float, float, str or None]:
    """
    Evaluate an inverse kinematics result.
    :param title: The inverse kinematics method which was used.
    :param pos: The position of the end effector.
    :param goal_pos: The position the end effector was trying to reach.
    :param quat:The orientation of the end effector.
    :param goal_quat: The orientation the end effector was trying to reach.
    :param duration: How long it took to compute the inverse kinematics.
    :param error: The acceptable error tolerance in meters and degrees of which to consider a solution successful.
    :param joints: The joints the model proposed.
    :param solution: A solution for joints to reach the target.
    :param verbose: If output messages should be logged or not.
    :return: If the move was successful or not, the position error, orientation error, and a potential error message.
    """
    orientation = quat is not None and goal_quat is not None
    # Get Euler angles so final difference is in a form easily understandable by a human.
    if orientation:
        euler = quaternion_to_euler(quat)
        goal_euler = quaternion_to_euler(goal_quat)
        diff_euler = np.linalg.norm(np.array(euler) - np.array(goal_euler))
    else:
        diff_euler = 0
    # Check if the position reaching was a success.
    diff_pos = np.linalg.norm(np.array(pos) - np.array(goal_pos))
    if diff_pos <= error:
        success = True
        diff_pos = 0
    else:
        success = False
    # If the position reached, also check the orientation if it was passed as well.
    if success and orientation:
        if diff_euler <= error:
            diff_euler = 0
        else:
            success = False
    # Create a message for the console if verbose.
    if verbose:
        s = f"{title} | {f'Success' if success else f'Failure'} | {neat(duration)} seconds"
        s += f"\nExpected = Position: [{neat(goal_pos[0])}, {neat(goal_pos[1])}, {neat(goal_pos[2])}]"
        if orientation:
            s += (f", Orientation: [{neat(goal_quat[0])}, {neat(goal_quat[1])}, {neat(goal_quat[2])}, "
                  f"{neat(goal_quat[3])}]")
        s += f"\nResults  = Position: [{neat(pos[0])}, {neat(pos[1])}, {neat(pos[2])}]"
        if orientation:
            s += f", Orientation: [{neat(quat[0])}, {neat(quat[1])}, {neat(quat[2])}, {neat(quat[3])}]"
        s += f"\nError    = Position: {neat(diff_pos)}"
        if orientation:
            s += f", Euler: {neat(diff_euler)}"
        print(s)
    # Create a success message to help improve results.
    if success:
        s = f"Successfully reached position [{neat(goal_pos[0])}, {neat(goal_pos[1])}, {neat(goal_pos[2])}]"
        if orientation:
            s += (f" and orientation [{neat(goal_quat[0])}, {neat(goal_quat[1])}, {neat(goal_quat[2])}, "
                  f"{neat(goal_quat[3])}]")
        s += "."
        if joints is not None and len(joints) > 0:
            multiple = len(joints) > 1
            s += f" The joint{'s' if multiple else ''} the method produced {'were' if multiple else 'was'} ["
            if len(joints) > 0:
                if isinstance(joints[0], float):
                    s += f"{neat(joints[0])}"
                else:
                    s += f"{joints[0]}"
            for i in range(1, len(joints)):
                if isinstance(joints[0], float):
                    s += f", {neat(joints[i])}"
                else:
                    s += f", {joints[i]}"
            s += f"]."
        else:
            s += " Did not produce any joints."
        return True, diff_pos, diff_euler, s
    # Create a failure message to help improve results.
    s = f"Failed to reach position [{neat(goal_pos[0])}, {neat(goal_pos[1])}, {neat(goal_pos[2])}]"
    if orientation:
        s += (f" and orientation [{neat(goal_quat[0])}, {neat(goal_quat[1])}, {neat(goal_quat[2])}, "
              f"{neat(goal_quat[3])}]")
    s += f". Instead reached position [{neat(pos[0])}, {neat(pos[1])}, {neat(pos[2])}]"
    if orientation:
        s += f" and orientation [{neat(quat[0])}, {neat(quat[1])}, {neat(quat[2])}, {neat(quat[3])}]"
    s += f"."
    if joints is not None and len(joints) > 0:
        s += f" The joints produced were ["
        if len(joints) > 0:
            if isinstance(joints[0], float):
                s += f"{neat(joints[0])}"
            else:
                s += f"{joints[0]}"
        for i in range(1, len(joints)):
            if isinstance(joints[0], float):
                s += f", {neat(joints[i])}"
            else:
                s += f", {joints[i]}"
        if solution is not None and len(solution) > 0:
            s += f"]. The solution for the joints were ["
            if len(solution) > 0:
                s += f"{neat(solution[0])}"
            for i in range(1, len(solution)):
                s += f", {neat(solution[i])}"
            s += "]."
        else:
            s += "]. There is no known solution to this, it may be unreachable."
    else:
        s += " Failed to produce any joints."
    return False, diff_pos, diff_euler, s


def test_ik(names: str or list or None = None, error: float = 0.001, orientation: bool = False, limits: bool = False,
            collisions: bool = False, verbose: bool = False, tests: int = 1, methods: str or list or None = None,
            targets: list or None = None, max_steps: int = 1000, tcp: int = 0) -> dict:
    """
    Test all inverse kinematics solvers for a model.
    :param names: The names of the robots to test.
    :param error: The acceptable error tolerance in meters and degrees of which to consider a solution successful.
    :param orientation: If orientation should be solved for.
    :param limits: If limits should be enforced or not.
    :param collisions: If collisions should be enforced or not.
    :param verbose: If output messages should be logged or not.
    :param tests: The number of tests to run.
    :param methods: Which methods to run.
    :param targets: Manually define targets.
    :param max_steps: How many iterations the DeepMind IK can run for.
    :param tcp: The index of the TCP we want.
    :return: The results in a dictionary.
    """
    # See if targets were manually defined.
    manual = targets is not None and len(targets) > 0
    if manual:
        tests = len(targets)
        orientation = True
        for target in targets:
            if len(target) < 2:
                orientation = False
                break
    # Need to check for methods which have solvers.
    solver_folder = os.path.join(os.getcwd(), "Solvers")
    # If no names were passed, try with all options.
    if names is None:
        names = []
        if os.path.exists(solver_folder):
            for name in os.listdir(solver_folder):
                names.append(name)
    # If one name was passed, convert it to a list.
    elif isinstance(names, str):
        names = [names]
    # If the methods to run was a string, make it into a list.
    if isinstance(methods, str):
        methods = [methods]
    # Need to ensure there are solvers that exist
    filtered = []
    for name in names:
        _, _, _, _, _, _, _, _, instance_tcp = load_model(name, orientation, limits, collisions, tcp)
        # Check depending on if we are solving for the entire transform or just the position.
        current_solver = os.path.join(solver_folder, name, "Transform" if orientation else "Position",
                                      str(instance_tcp))
        # Check for solvers.
        if os.path.exists(current_solver):
            for file in os.listdir(current_solver):
                # Ensure we are only checking Python files.
                if not file.endswith(".py"):
                    continue
                # Get the name from the file.
                module_name = file[:-3]
                # If it is not a method we are interested in, ignore it.
                if methods is not None and module_name not in methods:
                    continue
                # Load the code.
                # noinspection PyBroadException
                try:
                    spec = importlib.util.spec_from_file_location(module_name, os.path.join(current_solver, file))
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    # If it has a proper module, we can test this robot.
                    if hasattr(module, "inverse_kinematics"):
                        filtered.append(name)
                        break
                except Exception:
                    continue
    # Use the filtered results.
    names = filtered
    all_results = {}
    post = ""
    if orientation:
        post += " Orientation"
    if limits:
        post += " Limits"
    if collisions:
        post += " Collisions"
    # Loop for all robot options.
    index = 1
    for name in names:
        # Load the robot.
        model, data, lower, upper, site, path, solvers, joint_names, instance_tcp = load_model(name, orientation,
                                                                                               limits, collisions, tcp)
        if model is None or len(solvers) == 0:
            continue
        pre = "" if len(names) <= 1 else f"Model {index} / {len(names)} | "
        index += 1
        results = []
        n_joints = len(joint_names)
        for i in range(tests):
            if tests > 0:
                if verbose:
                    print()
                print(f"{pre}{name}{post} | Test {i + 1} / {tests}")
                if verbose:
                    print()
            result = {}
            # Define the starting pose at the middle.
            mid_positions(model, data, lower, upper)
            starting = get_joints(model, data, n_joints)
            quat = None
            if targets is None:
                # Determine where to move to.
                solution = starting
                # Ensure the starting position is not the same as the ending position.
                while solution == starting:
                    random_positions(model, data, lower, upper)
                    solution = get_joints(model, data, n_joints)
                pos, quat = get_pose(model, data, instance_tcp)
                if not orientation:
                    quat = None
            else:
                solution = None
                pos = targets[i][0]
                if orientation:
                    quat = targets[i][1]
            # Test the DeepMind inverse kinematics.
            duration = deepmind_ik(model, data, path, site, pos, quat, joint_names, max_steps)
            result_pos, result_quat = get_pose(model, data, instance_tcp)
            joints = get_joints(model, data, n_joints)
            success, error_pos, error_quat, message = eval_ik("DeepMind IK", result_pos, pos, result_quat, quat,
                                                              duration, error, joints, solution, verbose)
            # If DeepMind inverse kinematics was successful, check to see if it should be used as the solution.
            if success:
                # Check to see what joint configuration is the closest to the midpoints.
                existing_diff = 0
                new_diff = 0
                for j in range(len(starting)):
                    existing_diff += abs(starting[j] - solution[j])
                    new_diff += abs(starting[j] - joints[j])
                # If the DeepMind solution is closer to the midpoints, use it.
                if new_diff < existing_diff:
                    solution = joints
            result["DeepMind IK"] = {"Success": success, "Position": error_pos, "Orientation": orientation,
                                     "Duration": duration, "Message": message}
            # Use all solvers which were loaded.
            for solver in solvers:
                if methods is not None and solver["Name"] not in methods:
                    continue
                if verbose:
                    print()
                # Move back to the starting pose for every attempt.
                set_joints(model, data, starting)
                joints = []
                start_time = time.time()
                exception_message = None
                # Continue in case there are errors which still outputting the stacktrace for debugging.
                # noinspection PyBroadException
                try:
                    if orientation:
                        joints = solver["Method"]([pos[0], pos[1], pos[2]], [quat[0], quat[1], quat[2], quat[3]])
                    else:
                        joints = solver["Method"]([pos[0], pos[1], pos[2]])
                except Exception:
                    if verbose:
                        traceback.print_exc()
                    exception_message = traceback.format_exc()
                end_time = time.time()
                set_joints(model, data, joints)
                result_pos, result_quat = get_pose(model, data, instance_tcp)
                duration = end_time - start_time
                success, error_pos, error_quat, message = eval_ik(solver["Name"], result_pos, pos, result_quat, quat,
                                                                  end_time - start_time, error, joints, solution,
                                                                  verbose)
                if exception_message is not None:
                    message += f" This is likely due to the exception which happened: {exception_message}"
                result[solver["Name"]] = {"Success": success, "Position": error_pos, "Orientation": orientation,
                                          "Duration": duration, "Message": message}
            results.append(result)
        # If there is no results, there is nothing else to do.
        if len(results) < 1:
            return {}
        successes = {}
        durations = {}
        feedbacks = {}
        error_pos = {}
        error_quat = {}
        # Loop all results to get average scores.
        for result in results:
            result: dict
            for key in result.keys():
                success = result[key]["Success"]
                if key in successes:
                    durations[key] += result[key]["Duration"]
                    feedbacks[key] += f"\n{result[key]['Message']}"
                    if success:
                        successes[key] += 1
                    error_pos[key] += result[key]["Position"]
                    error_quat[key] += result[key]["Orientation"]
                else:
                    successes[key] = 1 if success else 0
                    durations[key] = result[key]["Duration"]
                    feedbacks[key] = result[key]["Message"]
                    error_pos[key] = result[key]["Position"]
                    error_quat[key] = result[key]["Orientation"]
        results = {}
        # Get messages in readable format.
        for key in successes:
            results[key] = {"Success": (successes[key] / tests * 100), "Position": (error_pos[key] / tests),
                            "Orientation": (error_quat[key] / tests), "Duration": (durations[key] / tests),
                            "Message": feedbacks[key]}
        # Sort the results by best to worst.
        results = dict(sorted(results.items(), key=lambda val: (
            -val[1]["Success"],
            val[1]["Position"],
            val[1]["Orientation"],
            val[1]["Duration"],
            val[0]
        )))
        # Append to messages to potentially help with improving results.
        core = "% success rate solving inverse kinematics"
        if not manual:
            core += (". All targets were reachable, and solutions to failures have been solved by another inverse "
                     "kinematics solver which have been provided")
        core += ":\n"
        for key in successes:
            trimmed = f"{results[key]['Success']:.2f}".rstrip('0').rstrip('.')
            if tests > 1:
                results[key]["Message"] = f"{trimmed}{core}{results[key]['Message']}"
        # Display the results.
        print(f"\n{pre}{name} TCP {instance_tcp}{post} | Results")
        for key in results.keys():
            s = (f"{key} | Success Rate = {neat(results[key]['Success'])}% | "
                 f"Position Error = {neat(results[key]['Position'])}")
            if orientation:
                s += f" | Orientation Error = {neat(results[key]['Orientation'])}"
            print(s + f" | Average Time = {neat(results[key]['Duration'])} seconds")
        # Optionally display the training methods.
        if verbose:
            print(f"\n{pre}{name} TCP {instance_tcp}{post} | Feedback")
            for key in results.keys():
                print(f"\n{key}\n{results[key]['Message']}")
        all_results[name] = results
    return all_results


def evaluate(names: str or list or None = None, error: float = 0.001, limits: bool = False, collisions: bool = False,
             tests: int = 1) -> None:
    """
    Evaluate robots for many trials.
    :param names: The names of the robots to test.
    :param error: The acceptable error tolerance in meters and degrees of which to consider a solution successful.
    :param limits: If tests with limits should be done as well.
    :param collisions: If tests with collisions should be done as well.
    :param tests: The number of tests to run.
    :return: Nothing.
    """
    # Ensure the folder for results exists.
    root = os.path.join(os.getcwd(), "Results")
    if not os.path.exists(root):
        os.mkdir(root)
    # Test all variations if requested.
    limits = [False, True] if limits else [False]
    collisions = [False, True] if collisions else [False]
    # Test all permutations of options.
    for orientation in [False, True]:
        for limit in limits:
            for collision in collisions:
                # Test the current permutation.
                results = test_ik(names, error, orientation, limit, collision, False, tests, None)
                # Write the results of each robot in CSV format.
                for robot in results.keys():
                    # Write the header.
                    s = f"Method,Success Rate(%),Position Error (m)"
                    if orientation:
                        s += ",Orientation (rad)"
                    s += f",Average Time (s)"
                    # Write all results.
                    for mode in results[robot].keys():
                        s += (f"\n{mode},{neat(results[robot][mode]['Success'])}%,"
                              f"{neat(results[robot][mode]['Position'])}")
                        if orientation:
                            s += f",{neat(results[robot][mode]['Orientation'])}"
                        s += f",{neat(results[robot][mode]['Duration'])}"
                    # Add the configuration details, so they are saved to the proper file.
                    if orientation:
                        robot += " Orientations"
                    if limit:
                        robot += " Limits"
                    if collision:
                        robot += " Collisions"
                    # Write to the file.
                    f = open(os.path.join(root, f"{robot}.csv"), "w")
                    f.write(s)
                    f.close()


def view(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """
    View the current Mujoco model.
    :param model: The Mujoco model.
    :param data: The data for the Mujoco model.
    :return: Nothing.
    """
    viewer = mujoco.viewer.launch_passive(model, data)
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
