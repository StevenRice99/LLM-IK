import math
import random

import mujoco
import mujoco.viewer
import numpy as np
import dm_control.utils.inverse_kinematics
import os
from dm_control import mujoco as mujoco_dm
from dm_control.mujoco.wrapper import mjbindings


def load_model(path):
    model = mujoco.MjModel.from_xml_path(path)
    lower = []
    upper = []
    for i in range(model.njnt):
        if model.jnt_limited[i]:
            lower.append(model.jnt_range[i][0])
            upper.append(model.jnt_range[i][1])
        else:
            lower.append(float('-inf'))
            upper.append(float('inf'))
    return model, mujoco.MjData(model), lower, upper, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite - 1)


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
    pos = [pos[0], pos[1], pos[2]]
    quat = [quat[0], quat[1], quat[2], quat[3]]
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


def main():
    path = os.path.join(os.getcwd(), "models", "universal_robots_ur5e", "ur5e.xml")

    model, data, lower, upper, site = load_model(path)

    #pos, angles = mid_positions(model, data, lower, upper)
    random_positions(model, data, lower, upper)
    goal_pos, goal_quat = get_pose(model, data)
    print(f"Goal   = {goal_pos} | {goal_quat}")

    random_positions(model, data, lower, upper)

    mujoco_ik(model, data, path, site, goal_pos, goal_quat)
    result_pos, result_quat = get_pose(model, data)
    print(f"Result = {result_pos} | {result_quat}")


if __name__ == "__main__":
    main()
