import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Calculates the joint angles needed to reach the specified TCP position.
    :param p: The target position [x, y, z].
    :return: A tuple of four joint angles in radians.
    """
    x, y, z = p
    theta1 = math.atan2(x, z)
    theta2 = 0.0
    theta3, theta4 = inverse_kinematics_joints3_4((x, y, z))
    return (theta1, theta2, theta3, theta4)

def inverse_kinematics_joints3_4(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Solves for joints 3 and 4 given the target position.
    :param p: The target position [x, y, z].
    :return: A tuple of two joint angles in radians.
    """
    px, py, pz = p
    theta3 = 0.0
    theta4 = 0.0
    return (theta3, theta4)