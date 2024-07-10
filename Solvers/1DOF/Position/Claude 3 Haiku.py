import numpy as np
from math import atan2, acos, pi

def inverse_kinematics(p: list) -> list:
    """
    Solve the inverse kinematics to reach the position in the reference frame of the chain's origin.
    :param p: The position for the end effector to reach in a Cartesian [x, y, z].
    :return: A list of the joint values in radians needed for the end effector to reach the position.
    """
    # Unpack the end effector position
    x, y, z = p

    # Calculate the joint angle
    theta = atan2(y, x)

    # Calculate the distance from the base to the end effector
    r = np.sqrt(x**2 + y**2)
    d = z - 0.4  # Subtract the end effector length

    # Solve for the joint position
    joint_pos = [theta]

    # Ensure the joint angle is within the valid range (-pi to pi)
    joint_pos[0] = (joint_pos[0] + 2*pi) % (2*pi)
    if joint_pos[0] > pi:
        joint_pos[0] -= 2*pi

    # Ensure the end effector can reach the desired position
    if r > 0.4:
        joint_pos[0] = np.arctan2(y, x)
    else:
        joint_pos[0] = 0

    # Ensure the joint angle is within the valid range (-pi/2 to pi/2)
    joint_pos[0] = max(-pi/2, min(joint_pos[0], pi/2))

    return joint_pos
