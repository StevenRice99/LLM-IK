import math
import random

import importlib
import importlib.util

import mujoco
import mujoco.viewer
import numpy as np
import dm_control.utils.inverse_kinematics
import os
from dm_control import mujoco as mujoco_dm
from dm_control.mujoco.wrapper import mjbindings


def load_model(name):
    path = os.path.join(os.getcwd(), "models", name, "model.xml")
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    lower = []
    upper = []
    joint_positions = []
    joint_orientations = []
    joint_axes = []
    previous_offset = [0, 0, 0]
    for i in range(model.njnt):
        body = model.jnt_bodyid[i]
        offset = model.jnt_pos[i]
        joint_positions.append(model.body_pos[body] - previous_offset + offset)
        previous_offset = offset
        joint_orientations.append(model.body_quat[body])
        joint_axes.append(model.jnt_axis[i])
        if model.jnt_limited[i]:
            lower.append(model.jnt_range[i][0])
            upper.append(model.jnt_range[i][1])
        else:
            lower.append(float('-inf'))
            upper.append(float('inf'))
    solvers = []
    solvers_directory = os.path.join(os.getcwd(), "solvers", name)
    for file in os.listdir(solvers_directory):
        if not file.endswith(".py"):
            continue
        module_name = file[:-3]
        module_path = os.path.join(solvers_directory, file)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        method = getattr(module, "inverse_kinematics")
        solvers.append({"Name": module_name, "Method": method})
    #joints_data = {}
    for i in range(model.njnt):
        pos = list(joint_positions[i])
        quat = list(joint_orientations[i])
        axis = list(joint_axes[i])
        #joint_data = {"Position": pos, "Orientation": quat, "Axis": axis}
        #if model.jnt_limited[i]:
        #    joint_data["Lower"] = lower[i]
        #    joint_data["Upper"] = upper[i]
        s = f"Joint {i + 1} = Position: {pos}, Orientation: {quat}, Axis: {axis}"
        if model.jnt_limited[i]:
            s += f", Lower: {lower[i]}, Upper: {upper[i]}"
        print(s)
        #joints_data[i + 1] = joint_data
    #print(joints_data)
    return model, data, lower, upper, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite - 1), path, solvers


def mid_positions(model, data, lower, upper):
    values = np.zeros(model.njnt)
    for i in range(model.njnt):
        values[i] = values[i] = (lower[i] + upper[i]) / 2
    return set_joints(model, data, values)


def random_positions(model, data, lower, upper):
    values = np.zeros(model.njnt)
    for i in range(model.njnt):
        values[i] = random.uniform(lower[i], upper[i])
    return set_joints(model, data, values)


def get_joints(model, data):
    values = []
    for i in range(model.njnt):
        values.append(data.qpos[i])
    return values


def set_joints(model, data, values):
    for i in range(model.njnt):
        data.ctrl[i] = values[i]
        data.qpos[i] = values[i]
        data.qvel[i] = 0
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)


def get_pose(model, data):
    pos = data.site_xpos[model.nsite - 1]
    raw = data.site_xmat[model.nsite - 1]
    quat = np.empty_like(data.xquat[model.nbody - 1])
    mjbindings.mjlib.mju_mat2Quat(quat, raw)
    quat /= quat.ptp()
    return [pos[0], pos[1], pos[2]], [quat[0], quat[1], quat[2], quat[3]]


def get_local_pose(pos, quat, data):
    return [pos[0] - data.xpos[0][0], pos[1] - data.xpos[0][1], pos[2] - data.xpos[0][2]], quaternion_difference(data.xquat[0], quat)


def quaternion_inverse(q):
    """
    Calculate the inverse of a quaternion.
    """
    w, x, y, z = q
    norm_sq = w**2 + x**2 + y**2 + z**2
    return [w / norm_sq, -x / norm_sq, -y / norm_sq, -z / norm_sq]


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return [w, x, y, z]


def quaternion_difference(q1, q2):
    """
    Calculate the difference between two quaternions.
    """
    q1_inv = quaternion_inverse(q1)
    relative_rotation = quaternion_multiply(q1_inv, q2)
    return relative_rotation


def quaternion_to_euler(quaternion):
    """
    Convert a quaternion to Euler angles (ZYX convention).
    """
    w, x, y, z = quaternion

    # ZYX convention
    # Roll (x-axis rotation)
    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

    sin_pitch = 2.0 * (w * y - z * x)

    # Pitch (y-axis rotation)
    # Use 90 degrees if out of range
    pitch = math.copysign(math.pi / 2, sin_pitch) if abs(sin_pitch) >= 1 else math.asin(sin_pitch)

    # Yaw (z-axis rotation)
    sin_yaw_cos_pitch = 2.0 * (w * z + x * y)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)

    return roll, pitch, yaw


def view(model, data):
    viewer = mujoco.viewer.launch_passive(model, data)
    while viewer.is_running():
        viewer.sync()


def mujoco_ik(model, data, path, site, pos, quat):
    physics = mujoco_dm.Physics.from_xml_path(path)
    copy_pos = pos.copy()
    copy_quat = quat.copy()
    values = dm_control.utils.inverse_kinematics.qpos_from_site_pose(physics, site, copy_pos, copy_quat)
    set_joints(model, data, values.qpos)


def test_ik(model, data, lower, upper, path, site, solvers, pos=None, quat=None, starting=None):
    if starting is None:
        random_positions(model, data, lower, upper)
        starting = get_joints(model, data)
    if pos is None or quat is None:
        random_positions(model, data, lower, upper)
        pos, quat = get_pose(model, data)
    print(f"Expected = {pos} | {quat}")
    view(model, data)
    pos_rel, quat_rel = get_local_pose(pos, quat, data)
    set_joints(model, data, starting)
    mujoco_ik(model, data, path, site, pos, quat)
    result_pos, result_quat = get_pose(model, data)
    print(f"Deepmind IK = {result_pos} | {result_quat}")
    #view(model, data)
    for solver in solvers:
        set_joints(model, data, starting)
        joints = solvers[0]["Method"](pos_rel[0], pos_rel[1], pos_rel[2], quat_rel[0], quat_rel[1], quat_rel[2], quat_rel[3])
        set_joints(model, data, joints)
        result_pos, result_quat = get_pose(model, data)
        print(f"{solver['Name']} = {result_pos} | {result_quat}")
        view(model, data)


def main():
    name = "universal_robots_ur5e"
    model, data, lower, upper, site, path, solvers = load_model(name)
    #test_ik(model, data, lower, upper, path, site, solvers)


if __name__ == "__main__":
    main()
