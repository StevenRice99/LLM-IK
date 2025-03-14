import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" using the Newton-Raphson method with pseudoinverse.
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    L2 = 0.093
    L3 = 0.09465
    theta1 = 0.0
    theta2 = 0.0
    theta3 = 0.0
    tolerance = 1e-06
    max_iterations = 1000
    damping_factor = 0.01
    for _ in range(max_iterations):
        fk_x = L2 * math.sin(theta2) * math.cos(theta3) + L3 * math.sin(theta3) * math.cos(theta1)
        fk_y = L2 * math.cos(theta2) * math.cos(theta3) + L3 * math.sin(theta3) * math.sin(theta1)
        fk_z = L2 * math.sin(theta2) * math.sin(theta3) + L3 * math.cos(theta3)
        error_x = fk_x - x
        error_y = fk_y - y
        error_z = fk_z - z
        J = np.array([[-L3 * math.sin(theta3) * math.sin(theta1), -L2 * math.cos(theta2) * math.cos(theta3) - L3 * math.sin(theta3) * math.cos(theta1), L2 * math.sin(theta2) * math.sin(theta3) + L3 * math.sin(theta3) * math.sin(theta1)], [L3 * math.sin(theta3) * math.cos(theta1), -L2 * math.sin(theta2) * math.cos(theta3) - L3 * math.cos(theta3) * math.sin(theta1), L2 * math.cos(theta2) * math.sin(theta3) + L3 * math.cos(theta3) * math.cos(theta1)], [0, L2 * math.cos(theta2) * math.sin(theta3) - L3 * math.sin(theta3), -L2 * math.sin(theta2) * math.cos(theta3) + L3 * math.sin(theta3)]])
        U, S, Vh = np.linalg.svd(J)
        S_inv = np.diag([1 / (s if s > damping_factor else damping_factor) for s in S])
        J_pinv = Vh.T @ S_inv @ U.T
        delta_theta = np.dot(J_pinv, np.array([error_x, error_y, error_z]))
        theta1 += delta_theta[0]
        theta2 += delta_theta[1]
        theta3 += delta_theta[2]
        if all((abs(err) < tolerance for err in [error_x, error_y, error_z])):
            break
    return (theta1, theta2, theta3)