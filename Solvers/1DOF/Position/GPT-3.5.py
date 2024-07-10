import numpy as np
from scipy.spatial.transform import Rotation

def inverse_kinematics(p: list) -> list:
    # Given parameters
    base_pos = np.array([0, 0, 0])
    base_ori = np.array([1, 0, 0, 0])  # Quaternion [w, x, y, z]

    joint1_pos = np.array([0, 0, 0])
    joint1_ori = np.array([1, 0, 0, 0])  # Quaternion [w, x, y, z]
    joint1_axis = np.array([1, 0, 0])   # Rotation axis for joint 1

    ee_pos = np.array([0, 0, 0.4])
    ee_ori = np.array([1, 0, 0, 0])     # Quaternion [w, x, y, z]

    # Target position in global coordinates
    target_pos = np.array(p)

    # Calculate r_0_e (vector from base to end effector in global coordinates)
    r_0_e = target_pos - base_pos

    # Calculate theta1 (angle for joint 1)
    theta1 = np.arctan2(r_0_e[1], r_0_e[0])

    # Transform r_0_e to joint 1's local coordinates
    r_1_e_local = Rotation.from_quat(joint1_ori).inv().apply(r_0_e)

    # Calculate theta2 (angle for joint 2)
    theta2 = np.arctan2(r_1_e_local[2], r_1_e_local[0])

    return [theta1, theta2]
