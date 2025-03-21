import math
import numpy as np

def euler_to_matrix(rx, ry, rz):
    """Converts Euler angles (roll, pitch, yaw) to a rotation matrix using ZYX convention."""
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    px, py, pz = p
    rx, ry, rz = r
    R_target = euler_to_matrix(rx, ry, rz)
    r31 = R_target[2, 0]
    r33 = R_target[2, 2]
    theta2 = math.atan2(-r31, r33)
    A = 0.425 * math.sin(theta2)
    B = 0.01615
    det = A ** 2 + B ** 2
    if abs(det) < 1e-06:
        raise ValueError('Determinant is zero, target is unreachable.')
    cos_theta1 = (A * px + B * py) / det
    sin_theta1 = (A * py - B * px) / det
    theta1 = math.atan2(sin_theta1, cos_theta1)
    return (theta1, theta2)