import math
from typing import List


def inverse_kinematics(p: List[float]) -> List[float]:
    """
    Solve the inverse kinematics to reach the position and orientation, both in the reference frame of the chain's origin.
    :param p: The position for the end effector to reach in a Cartesian [x, y, z].
    :return: A list of the joint values in radians needed for the end effector to reach the position.
    """
    # Extract y and z from the target position
    y = p[1]
    z = p[2]

    # Calculate the joint angle theta using atan2
    theta = math.atan2(-y, z)

    return [theta]
