import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(x, z)
    adjusted_x = x * math.cos(theta1) + z * math.sin(theta1)
    adjusted_z = -x * math.sin(theta1) + z * math.cos(theta1)
    distance = math.sqrt(adjusted_x ** 2 + adjusted_z ** 2)
    theta2 = math.acos(adjusted_x / distance)
    value = y / distance
    value = max(min(value, 1.0), -1.0)
    theta3 = math.asin(value)
    return (theta1, theta2, theta3)