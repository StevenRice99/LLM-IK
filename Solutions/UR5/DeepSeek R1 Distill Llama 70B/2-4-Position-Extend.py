import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    A = 0.39225
    B = 0.093
    C = 0.09465
    theta1 = math.atan2(x, z - C)
    theta2 = 0.0
    link2_x = A * math.cos(theta1)
    link2_z = A * math.sin(theta1) + C
    remaining_x = x - link2_x
    remaining_y = y - B
    remaining_z = z - link2_z
    theta3 = math.atan2(remaining_y, remaining_x)
    return (theta1, theta2, theta3)