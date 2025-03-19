import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta3 = 0.0
    joint3_x = x
    joint3_y = y
    joint3_z = z - 0.09465
    theta1 = math.atan2(joint3_x, joint3_z)
    r_joint3 = math.sqrt(joint3_x ** 2 + joint3_z ** 2)
    joint2_x = 0
    joint2_y = 0
    joint2_z = 0.39225
    v2_3_x = joint3_x - joint2_x
    v2_3_y = joint3_y - joint2_y
    v2_3_z = joint3_z - joint2_z
    d2_3 = math.sqrt(v2_3_x ** 2 + v2_3_y ** 2 + v2_3_z ** 2)
    cos_theta2 = v2_3_z / d2_3
    sin_theta2 = v2_3_x / d2_3
    theta2 = math.atan2(sin_theta2, cos_theta2)
    theta2 = theta2 - theta1
    return (theta1, theta2, theta3)