import math
import numpy as np

def forward_kinematics(theta1, theta2, theta3):
    """
    Computes the TCP position given the joint angles.
    :param theta1: First joint angle in radians.
    :param theta2: Second joint angle in radians.
    :param theta3: Third joint angle in radians.
    :return: TCP position as a tuple (x, y, z).
    """
    a1, a2, a3 = (0, 0.425, 0.39225)
    d_tcp = 0.093

    def transformation(theta, a, d):
        return np.array([[math.cos(theta), 0, math.sin(theta), a * math.cos(theta)], [math.sin(theta), 1, -math.cos(theta), a * math.sin(theta)], [0, 0, 1, d], [0, 0, 0, 1]])
    T1 = transformation(theta1, a1, 0)
    T2 = transformation(theta2, a2, 0)
    T3 = transformation(theta3, a3, 0)
    T_tcp = transformation(0, 0, d_tcp)
    T_total = np.dot(np.dot(np.dot(T1, T2), T3), T_tcp)
    x = T_total[0, 3]
    y = T_total[1, 3]
    z = T_total[2, 3]
    return (x, y, z)

def jacobian(theta1, theta2, theta3):
    """
    Computes the Jacobian matrix at the given joint angles.
    :param theta1: First joint angle in radians.
    :param theta2: Second joint angle in radians.
    :param theta3: Third joint angle in radians.
    :return: Jacobian matrix as a 3x3 numpy array.
    """
    a1, a2, a3 = (0, 0.425, 0.39225)
    d_tcp = 0.093

    def partial_derivative(theta1, theta2, theta3, joint):
        if joint == 1:
            dx_dtheta1 = -a2 * math.sin(theta1) * math.cos(theta2 + theta3) - a3 * math.sin(theta1) * math.cos(theta2)
            dy_dtheta1 = a2 * math.cos(theta1) * math.cos(theta2 + theta3) + a3 * math.cos(theta1) * math.cos(theta2)
            dz_dtheta1 = 0
        elif joint == 2:
            dx_dtheta2 = -a2 * math.sin(theta2) * math.cos(theta1) - a3 * math.sin(theta2 + theta3) * math.cos(theta1)
            dy_dtheta2 = a2 * math.cos(theta2) * math.cos(theta1) + a3 * math.cos(theta2 + theta3) * math.cos(theta1)
            dz_dtheta2 = a2 * math.cos(theta2) + a3 * math.cos(theta2 + theta3)
        elif joint == 3:
            dx_dtheta3 = -a3 * math.sin(theta3) * math.cos(theta1)
            dy_dtheta3 = a3 * math.cos(theta3) * math.cos(theta1)
            dz_dtheta3 = a3 * math.cos(theta3)
        else:
            dx_dtheta1, dy_dtheta1, dz_dtheta1 = (0, 0, 0)
        return np.array([dx_dtheta1, dy_dtheta1, dz_dtheta1])
    J = np.zeros((3, 3))
    J[:, 0] = partial_derivative(theta1, theta2, theta3, 1)
    J[:, 1] = partial_derivative(theta1, theta2, theta3, 2)
    J[:, 2] = partial_derivative(theta1, theta2, theta3, 3)
    return J

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" using iterative Jacobian method.
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x_target, y_target, z_target = p
    theta1 = math.atan2(x_target, z_target)
    theta2 = 0.0
    theta3 = 0.0
    tolerance = 0.001
    max_iterations = 1000
    lambda_ = 0.1
    for _ in range(max_iterations):
        x_current, y_current, z_current = forward_kinematics(theta1, theta2, theta3)
        error = np.array([x_target - x_current, y_target - y_current, z_target - z_current])
        error_norm = np.linalg.norm(error)
        if error_norm < tolerance:
            break
        J = jacobian(theta1, theta2, theta3)
        delta_theta = np.dot(np.linalg.pinv(J + lambda_ * np.eye(3)), error)
        theta1 += delta_theta[0]
        theta2 += delta_theta[1]
        theta3 += delta_theta[2]
    return (theta1, theta2, theta3)