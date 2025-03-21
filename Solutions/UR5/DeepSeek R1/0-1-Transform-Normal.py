import numpy as np

def rpy_to_matrix(roll, pitch, yaw):
    """Converts RPY angles to a rotation matrix using ZYX convention."""
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    x_target, y_target, z_target = p
    roll_target, pitch_target, yaw_target = r
    R_target = rpy_to_matrix(roll_target, pitch_target, yaw_target)
    B = 0.01615
    z_ratio = z_target / 0.425
    z_ratio = np.clip(z_ratio, -1.0, 1.0)
    theta2_1 = np.arccos(z_ratio)
    theta2_2 = -theta2_1
    for theta2 in [theta2_1, theta2_2]:
        A = 0.425 * np.sin(theta2)
        denominator = A ** 2 + B ** 2
        if np.isclose(denominator, 0, atol=1e-06):
            continue
        cos_theta1 = (A * x_target + B * y_target) / denominator
        sin_theta1 = (-B * x_target + A * y_target) / denominator
        norm = np.hypot(sin_theta1, cos_theta1)
        if norm < 1e-06:
            continue
        sin_theta1 /= norm
        cos_theta1 /= norm
        theta1 = np.arctan2(sin_theta1, cos_theta1)
        R_z = np.array([[np.cos(theta1), -np.sin(theta1), 0], [np.sin(theta1), np.cos(theta1), 0], [0, 0, 1]])
        R_y = np.array([[np.cos(theta2), 0, np.sin(theta2)], [0, 1, 0], [-np.sin(theta2), 0, np.cos(theta2)]])
        R_robot = R_z @ R_y
        if np.allclose(R_robot, R_target, atol=1e-06) or np.allclose(R_robot, -R_target, atol=1e-06):
            return (theta1, theta2)
    theta2 = theta2_1
    A = 0.425 * np.sin(theta2)
    denominator = A ** 2 + B ** 2
    if np.isclose(denominator, 0, atol=1e-06):
        theta1 = 0.0
    else:
        cos_theta1 = (A * x_target + B * y_target) / denominator
        sin_theta1 = (-B * x_target + A * y_target) / denominator
        theta1 = np.arctan2(sin_theta1, cos_theta1)
    return (theta1, theta2)