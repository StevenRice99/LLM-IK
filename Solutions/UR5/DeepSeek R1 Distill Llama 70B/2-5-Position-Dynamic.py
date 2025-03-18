import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    theta1, theta2 = inverse_kinematics_joints_1_2(p)
    x, y, z = p
    adjusted_x = x - 0.39225 * math.sin(theta1)
    adjusted_z = z - 0.39225 * math.cos(theta1)
    theta3, theta4 = inverse_kinematics_joints_3_4((adjusted_x, y, adjusted_z))
    return (theta1, theta2, theta3, theta4)

def inverse_kinematics_joints_1_2(p: tuple[float, float, float]) -> tuple[float, float]:
    x, y, z = p
    theta1 = math.atan2(x, z)
    adjusted_x = x - 0.39225 * math.sin(theta1)
    adjusted_z = z - 0.39225 * math.cos(theta1)
    theta_total = math.atan2(adjusted_x, adjusted_z)
    theta2 = theta_total - theta1
    return (theta1, theta2)

def inverse_kinematics_joints_3_4(p: tuple[float, float, float]) -> tuple[float, float]:
    x, y, z = p
    theta3 = math.atan2(-x, y)
    theta4 = 0.0
    return (theta3, theta4)