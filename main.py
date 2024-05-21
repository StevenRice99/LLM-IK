import math
import random

import mujoco
import mujoco.viewer
import numpy as np
import os


def load_model(path):
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    return model, data


def get_joint_limits(model):
    #joint_names = []
    lower = []
    upper = []
    for i in range(model.njnt):
        #joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))
        if model.jnt_limited[i]:
            lower.append(model.jnt_range[i][0])
            upper.append(model.jnt_range[i][1])
        else:
            lower.append(float('-inf'))
            upper.append(float('inf'))
    return lower, upper


def mid_positions(model, data, lower, upper, stopping: float = 0.00000001):
    joint_positions = np.zeros(model.njnt)
    for i in range(model.njnt):
        joint_positions[i] = joint_positions[i] = (lower[i] + upper[i]) / 2
    return set_joint_positions(model, data, joint_positions, stopping)


def random_positions(model, data, lower, upper, stopping: float = 0.00000001):
    joint_positions = np.zeros(model.njnt)
    for i in range(model.njnt):
        joint_positions[i] = random.uniform(lower[i], upper[i])
    return set_joint_positions(model, data, joint_positions, stopping)


def set_joint_positions(model, data, joint_positions, stopping: float = 0.00000001):
    for i in range(model.njnt):
        data.ctrl[i] = joint_positions[i]
        #data.qpos[i] = joint_positions[i]
        data.qvel[i] = 0
    mujoco.mj_forward(model, data)
    moving = True
    while moving:
        mujoco.mj_step(model, data)
        moving = False
        for i in range(model.njnt):
            if data.qvel[i] > stopping:
                moving = True
                break
    pos = data.xpos[model.nbody - 1] - data.xpos[0]
    difference = quaternion_difference(data.xquat[0], data.xquat[model.nbody - 1])
    angles = quaternion_to_euler(difference)
    return pos, angles


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
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def main():
    path = os.path.join(os.getcwd(), "models", "universal_robots_ur5e", "ur5e.xml")

    model, data = load_model(path)

    lower, upper = get_joint_limits(model)

    #mid_positions(model, data, lower, upper)
    pos, angles = random_positions(model, data, lower, upper)
    print(f"Position Offset {pos}")
    print(f"Rotation Offset {angles}")

    viewer = mujoco.viewer.launch_passive(model, data)

    while viewer.is_running():
        viewer.sync()


if __name__ == "__main__":
    main()
