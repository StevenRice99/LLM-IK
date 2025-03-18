import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" using an iterative Jacobian method.
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x_target, y_target, z_target = p
    L1 = 0.4412
    L2 = 0.39225
    theta1 = 0.0
    theta2 = 0.0
    max_iterations = 1000
    error_threshold = 0.001
    damping = 0.1
    for _ in range(max_iterations):
        theta_sum = theta1 + theta2
        x_current = L1 * math.sin(theta1) + L2 * math.sin(theta_sum)
        z_current = L1 * math.cos(theta1) + L2 * math.cos(theta_sum)
        error = math.sqrt((x_current - x_target) ** 2 + (z_current - z_target) ** 2)
        if error < error_threshold:
            break
        J11 = L1 * math.cos(theta1) + L2 * math.cos(theta_sum)
        J12 = L2 * math.cos(theta_sum)
        J21 = -L1 * math.sin(theta1) - L2 * math.sin(theta_sum)
        J22 = -L2 * math.sin(theta_sum)
        det = J11 * J22 - J12 * J21
        det_damped = det ** 2 + damping
        J_inv11 = J22 * det_damped / det_damped
        J_inv12 = -J12 * det_damped / det_damped
        J_inv21 = -J21 * det_damped / det_damped
        J_inv22 = J11 * det_damped / det_damped
        e_x = x_target - x_current
        e_z = z_target - z_current
        delta_theta1 = J_inv11 * e_x + J_inv12 * e_z
        delta_theta2 = J_inv21 * e_x + J_inv22 * e_z
        theta1 += delta_theta1
        theta2 += delta_theta2
    return (theta1, theta2)