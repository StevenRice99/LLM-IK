import math

def inverse_kinematics(p: list) -> list:
    """
    Solve the inverse kinematics to reach the position in the reference frame of the chain's origin.
    :param p: The position for the end effector to reach in a Cartesian [x, y, z].
    :return: A list of the joint values in radians needed for the end effector to reach the position.
    """
    sx, sy, sz = p
    ax, ay = 0, 0

    # Calculate theta1
    theta1 = math.atan2(sy, sx)

    # Calculate theta2
    denom = math.sqrt(sx**2 + sy**2)
    theta2 = math.atan2(sz, denom) - math.atan2(ay, ax)

    return [theta1, theta2]
