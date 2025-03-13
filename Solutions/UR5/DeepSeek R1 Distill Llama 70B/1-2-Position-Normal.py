import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" using the Newton-Raphson method.
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x_target, y_target, z_target = p
    L1 = 0.425
    L2 = 0.39225
    theta1 = 0.0
    theta2 = 0.0
    tolerance = 1e-06
    max_iterations = 1000
    for _ in range(max_iterations):
        x_current = L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2)
        z_current = L1 * math.sin(theta1) + L2 * math.sin(theta1 + theta2)
        error_x = x_current - x_target
        error_z = z_current - z_target
        J11 = -L1 * math.sin(theta1) - L2 * math.sin(theta1 + theta2)
        J12 = -L2 * math.sin(theta1 + theta2)
        J21 = L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2)
        J22 = L2 * math.cos(theta1 + theta2)
        det_J = J11 * J22 - J12 * J21
        if abs(det_J) < 1e-12:
            break
        delta_theta1 = (J22 * error_x - J12 * error_z) / det_J
        delta_theta2 = (-J21 * error_x + J11 * error_z) / det_J
        theta1 += delta_theta1
        theta2 += delta_theta2
        if abs(error_x) < tolerance and abs(error_z) < tolerance:
            break
    return (theta1, theta2)