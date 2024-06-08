import math
import random
import importlib
import importlib.util
import time
import traceback
import mujoco
import mujoco.viewer
import numpy as np
import dm_control.utils.inverse_kinematics
import os
from dm_control import mujoco as mujoco_dm
from dm_control.mujoco.wrapper import mjbindings


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


def get_data(model: mujoco.MjModel, limits: bool = False) -> [mujoco.MjData, list, list, list, list, list, str]:
    """
    Get the data for the Mujoco model.
    :param model: The Mujoco model.
    :param limits: If limits should be enforced or not.
    :return: The data of the Mujoco model, joint positions, joint orientations, joint axes, joint lower bounds, joint
    upper bounds, and the name of the site for the end effector.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    positions = []
    orientations = []
    axes = []
    lower = []
    upper = []
    # No previous offset for the first joint.
    previous_offset = [0, 0, 0]
    # Loop every joint.
    for i in range(model.njnt):
        body = model.jnt_bodyid[i]
        # Some joints may be offset from their body, so we need to account for that.
        offset = model.jnt_pos[i]
        # Get this position taking into account how much the previous joint was offset.
        positions.append(model.body_pos[body] - previous_offset + offset)
        # Set the offset for the next iteration.
        previous_offset = offset
        orientations.append(model.body_quat[body])
        axes.append(model.jnt_axis[i])
        # Get limits.
        if model.jnt_limited[i]:
            # If we are not looking to enforce limits, remove them.
            if not limits:
                model.jnt_limited[i] = 0
                model.jnt_range[i][0] = -6.28319
                model.jnt_range[i][1] = 6.28319
            lower.append(model.jnt_range[i][0])
            upper.append(model.jnt_range[i][1])
        else:
            lower.append(-6.28319)
            upper.append(6.28319)
    return (data, positions, orientations, axes, lower, upper,
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite - 1))


def generate_prompt(name: str, orientation: bool = False, limits: bool = False) -> str:
    """
    Generate the prompt to give to LLMs.
    :param name: The name of the robot to generate the prompt for.
    :param orientation: If orientation should be solved for.
    :param limits: If limits should be enforced or not.
    :return: The prompt for the LLM.
    """
    model, path = import_model(name)
    if path is None:
        return ""
    data, positions, orientations, axes, lower, upper, site = get_data(model, limits)
    # Get the start of the prompt.
    with open(os.path.join(os.getcwd(), "Prompts", "prompt_start.txt"), 'r') as file:
        details = file.read()
    if limits:
        details += "Limits are in radians."
    # Write the base nicely.
    pos = data.xpos[1]
    pos = f"[{pos[0]:g}, {pos[1]:g}, {pos[2]:g}]"
    details += f"\n\nBase = Position: {pos}"
    quat = data.xquat[1]
    quat = f"[{quat[0]:g}, {quat[1]:g}, {quat[2]:g}, {quat[3]:g}]"
    details += f", Orientation: {quat}"
    # Write all joints nicely.
    for i in range(model.njnt):
        pos = positions[i]
        pos = f"[{pos[0]:g}, {pos[1]:g}, {pos[2]:g}]"
        details += f"\nJoint {i + 1} = Position: {pos}"
        quat = orientations[i]
        quat = f"[{quat[0]:g}, {quat[1]:g}, {quat[2]:g}, {quat[3]:g}]"
        details += f", Orientation: {quat}"
        axis = axes[i]
        axis = f"[{axis[0]:g}, {axis[1]:g}, {axis[2]:g}]"
        details += f", Axes: {axis}"
        if limits and model.jnt_limited[i]:
            details += f", Lower: {lower[i]:g}, Upper: {upper[i]:g}"
    # Write the end effector nicely.
    site_id = model.nsite - 1
    body = model.site_bodyid[site_id]
    pos = model.body_pos[body]
    pos = pos + model.site_pos[site_id]
    pos = f"[{pos[0]:g}, {pos[1]:g}, {pos[2]:g}]"
    quat = model.site_quat[site_id]
    quat = f"[{quat[0]:g}, {quat[1]:g}, {quat[2]:g}, {quat[3]:g}]"
    details += f"\nEnd Effector = Position: {pos}, Orientation: {quat}\n\n"
    # Write the remainder of the prompt.
    file = "prompt_end_transform.txt" if orientation else "prompt_end_position.txt"
    with open(os.path.join(os.getcwd(), "Prompts", file), 'r') as file:
        details += file.read()
    return details


def load_model(name: str, orientation: bool = False,
               limits: bool = False) -> [mujoco.MjModel, mujoco.MjData, list, list, str, str, dict]:
    """
    Load a model into Mujoco.
    :param name: The name of the file to load.
    :param orientation: If orientation should be solved for.
    :param limits: If limits should be enforced or not.
    :return: The Mujoco model, the data of the Mujoco model, joint lower bounds, joint upper bounds, the name of the
    site for the end effector, the path to the model, and all LLM solvers.
    """
    # Do regular Mujoco set up and get joints and other data we need.
    model, path = import_model(name)
    if model is None:
        return [None, None, None, None, None, None, path, None]
    data, positions, orientations, axes, lower, upper, site = get_data(model, limits)
    # Load in solvers which exist for this robot.
    solvers = []
    folder = "Transform" if orientation else "Position"
    solvers_directory = os.path.join(os.getcwd(), "Solvers", name, folder)
    if os.path.exists(solvers_directory):
        for file in os.listdir(solvers_directory):
            # Ensure we are only checking Python files.
            if not file.endswith(".py"):
                continue
            # Keep the name of the file and bind the inverse kinematics method.
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
    return model, data, lower, upper, site, path, solvers


def set_joints(model: mujoco.MjModel, data: mujoco.MjData, values: list) -> None:
    """
    Set the Mujoco model to the specified joint values.
    :param model: The Mujoco model.
    :param data: The data for the Mujoco model.
    :param values: The joint values to set.
    :return: Nothing.
    """
    number = min(model.njnt, len(values))
    for i in range(number):
        data.ctrl[i] = values[i]
        data.qpos[i] = values[i]
        data.qvel[i] = 0
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)


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


def get_joints(model: mujoco.MjModel, data: mujoco.MjData) -> list:
    """
    Get the joint values of the Mujoco model.
    :param model: The Mujoco model.
    :param data: The data for the Mujoco model.
    :return: The joint values of the Mujoco model.
    """
    values = []
    for i in range(model.njnt):
        values.append(data.qpos[i])
    return values


def get_pose(model: mujoco.MjModel, data: mujoco.MjData) -> [list, list]:
    """
    Get the current pose of the position and orientation of the end effector of the Mujoco model.
    :param model: The Mujoco model.
    :param data: The data for the Mujoco model.
    :return: The position [X, Y, Z] and the orientation [W, X, Y, Z].
    """
    pos = data.site_xpos[model.nsite - 1]
    raw = data.site_xmat[model.nsite - 1]
    quat = np.empty_like(data.xquat[model.nbody - 1])
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
                quat: list or None = None) -> float:
    """
    Run the inverse kinematics from Deepmind Controls.
    :param model: The Mujoco model.
    :param data: The data for the Mujoco model.
    :param path: The path for the Mujoco model to reload the model in the Deepmind Controls instance.
    :param site: The name of the site or end effector.
    :param pos: The position for the end effector to reach.
    :param quat: The orientation for the end effector to reach.
    :return: The time to took to complete the inverse kinematics.
    """
    physics = mujoco_dm.Physics.from_xml_path(path)
    # Copy positions as it seems they might get modified during use.
    pos_copy = pos.copy()
    quat_copy = None if quat is None else quat.copy()
    start_time = time.time()
    values = dm_control.utils.inverse_kinematics.qpos_from_site_pose(physics, site, pos_copy, quat_copy)
    end_time = time.time()
    set_joints(model, data, values.qpos)
    return end_time - start_time


def eval_ik(title: str, pos: list, goal_pos: list, quat: list or None = None, goal_quat: list or None = None,
            duration: float = 0, error: float = 0.001, joints: list or None = None, solution: list or None = None,
            verbose: bool = False) -> [bool, str or None]:
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
    :return: If the move was successful or not and a potential error message.
    """
    # Calculate position differences.
    diff_pos = [abs(pos[0] - goal_pos[0]), abs(pos[1] - goal_pos[1]), abs(pos[2] - goal_pos[2])]
    # Get Euler angles so final difference is in a form easily understandable by a human.
    if quat is not None and goal_quat is not None:
        euler = quaternion_to_euler(quat)
        goal_euler = quaternion_to_euler(goal_quat)
        diff_euler = [abs(euler[0] - goal_euler[0]), abs(euler[1] - goal_euler[1]), abs(euler[2] - goal_euler[2])]
    else:
        euler = [0, 0, 0]
        goal_euler = [0, 0, 0]
        diff_euler = [0, 0, 0]
    # Check if the position reaching was a success.
    success = np.linalg.norm(np.array(pos) - np.array(goal_pos)) <= error
    # If the position reached, also check the orientation if it was passed as well.
    if success and quat is not None and goal_quat is not None:
        success = np.linalg.norm(np.array(euler) - np.array(goal_euler)) <= error
    orientation = quat is not None and goal_quat is not None
    # Create a message for the console if verbose.
    if verbose:
        s = f"{title} | {f'Success' if success else f'Failure'} | {duration} seconds"
        s += f"\nExpected = Position: [{goal_pos[0]:g}, {goal_pos[1]:g}, {goal_pos[2]:g}]"
        if orientation:
            s += f", Orientation: [{goal_quat[0]:g}, {goal_quat[1]:g}, {goal_quat[2]:g}, {goal_quat[3]:g}]"
        s += f"\nResults  = Position: [{pos[0]:g}, {pos[1]:g}, {pos[2]:g}]"
        if orientation:
            s += f", Orientation: [{quat[0]:g}, {quat[1]:g}, {quat[2]:g}, {quat[3]:g}]"
        s += f"\nError    = Position: [{diff_pos[0]:g}, {diff_pos[1]:g}, {diff_pos[2]:g}]"
        if orientation:
            s += f", Euler: [{diff_euler[0]:g}, {diff_euler[1]:g}, {diff_euler[2]:g}]"
        # For now, just log the information. In the future if results are promising we would return and tabulate data.
        print(s)
    # Create a success message to help improve results.
    if success:
        s = f"Successfully reached position [{goal_pos[0]:g}, {goal_pos[1]:g}, {goal_pos[2]:g}]"
        if orientation:
            s += f" and orientation [{goal_quat[0]:g}, {goal_quat[1]:g}, {goal_quat[2]:g}, {goal_quat[3]:g}]"
        s += "."
        if joints is not None:
            s += f" The joints the method produced were ["
            if len(joints) > 0:
                s += f"{joints[0]:g}"
            for i in range(1, len(joints)):
                s += f", {joints[i]}"
            s += f"]."
            return True, s
    # Create a failure message to help improve results.
    s = f"Failed to reach position [{goal_pos[0]:g}, {goal_pos[1]:g}, {goal_pos[2]:g}]"
    if orientation:
        s += f" and orientation [{goal_quat[0]:g}, {goal_quat[1]:g}, {goal_quat[2]:g}, {goal_quat[3]:g}]"
    s += f". Instead reached position [{pos[0]:g}, {pos[1]:g}, {pos[2]:g}]"
    if orientation:
        s += f" and orientation [{quat[0]:g}, {quat[1]:g}, {quat[2]:g}, {quat[3]:g}]"
    s += f"."
    if joints is not None and solution is not None:
        s += f" The joints produced were ["
        if len(joints) > 0:
            s += f"{joints[0]:g}"
        for i in range(1, len(joints)):
            s += f", {joints[i]}"
        s += f"]. The solution for the joints were ["
        if len(solution) > 0:
            s += f"{solution[0]:g}"
        for i in range(1, len(solution)):
            s += f", {solution[i]}"
        s += "]."
    return False, s


def test_ik(name: str, error: float = 0.001, orientation: bool = False, limits: bool = False,
            verbose: bool = False, tests: int = 1, methods: list or None = None) -> dict:
    """
    Test all inverse kinematics solvers for a model.
    :param name: The name of the robot to test.
    :param error: The acceptable error tolerance in meters and degrees of which to consider a solution successful.
    :param orientation: If orientation should be solved for.
    :param limits: If limits should be enforced or not.
    :param verbose: If output messages should be logged or not.
    :param tests: The number of tests to run.
    :param methods: Which methods to run.
    :return: The results in a dictionary.
    """
    results = []
    # Load the robot.
    model, data, lower, upper, site, path, solvers = load_model(name, orientation, limits)
    if model is None:
        return {}
    for i in range(tests):
        if tests > 0:
            if verbose:
                print()
            print(f"{i+1} / {tests}")
            if verbose:
                print()
        result = {}
        # Determine where to move to.
        random_positions(model, data, lower, upper)
        solution = get_joints(model, data)
        pos, quat = get_pose(model, data)
        if not orientation:
            quat = None
        # Define the starting pose at the middle.
        mid_positions(model, data, lower, upper)
        starting = get_joints(model, data)
        # Test the Deepmind inverse kinematics.
        if methods is None or "Deepmind IK" in methods:
            duration = deepmind_ik(model, data, path, site, pos, quat)
            result_pos, result_quat = get_pose(model, data)
            joints = get_joints(model, data)
            success, message = eval_ik("Deepmind IK", result_pos, pos, result_quat, quat, duration, error, joints,
                                       solution, verbose)
            if success:
                solution = joints
            result["Deepmind IK"] = {"Success": success, "Message": message, "Duration": duration}
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
                    joints = solvers[0]["Method"]([pos[0], pos[1], pos[2]], [quat[0], quat[1], quat[2], quat[3]])
                else:
                    joints = solvers[0]["Method"]([pos[0], pos[1], pos[2]])
            except Exception:
                if verbose:
                    traceback.print_exc()
                exception_message = traceback.format_exc()
            end_time = time.time()
            set_joints(model, data, joints)
            result_pos, result_quat = get_pose(model, data)
            duration = end_time - start_time
            success, message = eval_ik(solver["Name"], result_pos, pos, result_quat, quat, end_time - start_time, error,
                                       joints, solution, verbose)
            if exception_message is not None:
                message += f" This is likely due to the exception which happened: {exception_message}"
            result[solver["Name"]] = {"Success": success, "Message": message, "Duration": duration}
        results.append(result)
    # If there is no results, there is nothing else to do.
    if len(results) < 1:
        return {}
    successes = {}
    durations = {}
    feedbacks = {}
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
            else:
                successes[key] = 1 if success else 0
                durations[key] = result[key]["Duration"]
                feedbacks[key] = result[key]["Message"]
    results = {}
    # Get messages in readable format.
    for key in successes:
        results[key] = {"Success": (successes[key] / tests * 100), "Duration": (durations[key] / tests),
                        "Message": feedbacks[key]}
    # Append to messages to potentially help with improving results.
    for key in successes:
        trimmed = f"{results[key]['Success']:.2f}".rstrip('0').rstrip('.')
        results[key]["Message"] = (f"The method had a success rate of {trimmed}% solving inverse kinematics. "
                                   f"Below are feedback messages of the test trails to analyze to improve the method:\n"
                                   f"{results[key]['Message']}")
    # Display the results.
    print("\nResults:")
    for key in results.keys():
        trimmed = f"{results[key]['Success']:.2f}".rstrip('0').rstrip('.')
        print(f"{key} | Success Rate = {trimmed}% | "
              f"Average Time = {results[key]['Duration']:f} seconds")
    # Optionally display the training methods.
    if verbose:
        print("\nFeedback:")
        for key in results.keys():
            print(f"\n{key}\n{results[key]['Message']}")
    return results


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