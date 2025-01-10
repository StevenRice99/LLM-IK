import math

def inverse_kinematics(p: tuple[float, float, float]) -> float:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: The value to set the link to for reaching position "p".
    """
    # Extract the x and y components of the target position
    x, y, z = p

    # Compute the angle theta using atan2 and negate it to match the expected orientation
    theta = -math.atan2(x, y)
    
    return theta