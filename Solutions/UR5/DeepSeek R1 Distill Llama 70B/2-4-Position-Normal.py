import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Computes the joint values needed to reach the target position "p" using the Jacobian pseudo-inverse method.
    :param p: The target position [x, y, z].
    :return: A tuple of joint values [q1, q2, q3].
    """
    q = np.array([0.0, 0.0, 0.0], dtype=float)
    target = np.array(p, dtype=float)
    max_iter = 1000
    tol = 1e-06
    lr = 0.01
    min_lr = 0.001
    for _ in range(max_iter):
        current_pos = forward_kinematics(q)
        error = target - current_pos
        error_norm = np.linalg.norm(error)
        if error_norm < tol:
            break
        J = jacobian(q)
        J_transpose = J.T
        reg = 1e-07
        J_pinv = np.linalg.inv(J @ J_transpose + reg * np.eye(3)) @ J_transpose
        dq = lr * J_pinv @ error
        q += dq
        q = apply_joint_limits(q)
        prev_error = error_norm
        q += dq
        q = apply_joint_limits(q)
        current_pos = forward_kinematics(q)
        error = target - current_pos
        new_error_norm = np.linalg.norm(error)
        if new_error_norm < prev_error:
            lr = min(lr * 1.1, 0.1)
        else:
            lr = max(lr * 0.9, min_lr)
    return (q[0], q[1], q[2])

def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """
    Computes the TCP position given the joint values.
    :param q: Joint values [q1, q2, q3].
    :return: TCP position [x, y, z].
    """
    q1, q2, q3 = q
    l1 = 0.39225
    l2 = 0.093
    offset = 0.09465
    theta = q1 + q2
    x = l2 * math.cos(theta) + l1 * math.cos(theta + q3)
    y = l2 * math.sin(theta) + l1 * math.sin(theta + q3)
    z = l1 * math.cos(q3) + offset
    return np.array([x, y, z])

def jacobian(q: np.ndarray) -> np.ndarray:
    """
    Computes the Jacobian matrix for the given joint values.
    :param q: Joint values [q1, q2, q3].
    :return: 3x3 Jacobian matrix.
    """
    q1, q2, q3 = q
    l1 = 0.39225
    l2 = 0.093
    theta = q1 + q2
    dx_dq1 = -l2 * math.sin(theta) - l1 * math.sin(theta + q3)
    dx_dq2 = -l2 * math.sin(theta)
    dx_dq3 = -l1 * math.sin(theta + q3)
    dy_dq1 = l2 * math.cos(theta) + l1 * math.cos(theta + q3)
    dy_dq2 = l2 * math.cos(theta)
    dy_dq3 = l1 * math.cos(theta + q3)
    dz_dq1 = 0.0
    dz_dq2 = 0.0
    dz_dq3 = -l1 * math.sin(q3)
    return np.array([[dx_dq1, dx_dq2, dx_dq3], [dy_dq1, dy_dq2, dy_dq3], [dz_dq1, dz_dq2, dz_dq3]])

def apply_joint_limits(q: np.ndarray) -> np.ndarray:
    """
    Ensures joint values stay within their limits using a smooth projection.
    :param q: Joint values to check.
    :return: Joint values within limits.
    """
    limits = np.array([[-6.2831853, 6.2831853], [-6.2831853, 6.2831853], [-6.2831853, 6.2831853]])
    for i in range(3):
        if q[i] < limits[i][0]:
            q[i] = limits[i][0]
        elif q[i] > limits[i][1]:
            q[i] = limits[i][1]
    return q