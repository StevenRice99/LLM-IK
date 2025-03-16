import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x_target, y_target, z_target = p
    theta1_range = (-math.pi, math.pi)
    theta2_range = (-math.pi, math.pi)
    theta3_range = (-math.pi, math.pi)
    step_size = 0.1
    min_distance = float('inf')
    best_angles = (0.0, 0.0, 0.0)
    for theta1 in range(int(theta1_range[0] / step_size), int(theta1_range[1] / step_size)):
        theta1 = theta1 * step_size
        for theta2 in range(int(theta2_range[0] / step_size), int(theta2_range[1] / step_size)):
            theta2 = theta2 * step_size
            for theta3 in range(int(theta3_range[0] / step_size), int(theta3_range[1] / step_size)):
                theta3 = theta3 * step_size
                x, y, z = FORWARD_KINEMATICS(joint1=theta1, joint2=theta2, joint3=theta3)
                distance = math.sqrt((x - x_target) ** 2 + (y - y_target) ** 2 + (z - z_target) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    best_angles = (theta1, theta2, theta3)
    return best_angles