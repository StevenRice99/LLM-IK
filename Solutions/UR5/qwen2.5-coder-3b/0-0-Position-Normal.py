import math

def inverse_kinematics(p: tuple[float, float, float]) -> float:
    """
    Gets the joint values needed to reach position "p".
    :param p :The position to reach in the form [x, y, z].
    :return: The value to set the link to for reaching position "p".
    """
    x_tcp, y_tcp, z_tcp = p
    
    # Calculate the distance from the base to the TCP
    d = math.sqrt(x_tcp**2 + y_tcp**2)
    
    # Calculate the joint angle using arctan2
    theta = math.atan2(y_tcp, x_tcp)
    
    # Ensure the angle is within the specified limits
    if theta < -math.pi:
        theta += 2 * math.pi
    elif theta > math.pi:
        theta -= 2 * math.pi
    
    return theta