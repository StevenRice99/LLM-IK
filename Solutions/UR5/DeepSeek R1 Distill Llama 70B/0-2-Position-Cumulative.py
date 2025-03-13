import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Computes the joint angles needed to reach the given position p.
    
    Args:
        p: A tuple (x, y, z) representing the target position.
    
    Returns:
        A tuple (theta1, theta2, theta3) in radians.
    """
    x, y, z = p
    z_max = 0.425
    cos_theta2 = z / z_max
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    sin_theta2 = math.sqrt(1 - cos_theta2 ** 2)
    E = 0.425 * sin_theta2
    denominator = E ** 2 + 0.01615 ** 2
    cos_theta1 = (E * x + 0.01615 * y) / denominator
    sin_theta1 = (E * y - 0.01615 * x) / denominator
    norm = math.sqrt(cos_theta1 ** 2 + sin_theta1 ** 2)
    cos_theta1 /= norm
    sin_theta1 /= norm
    theta1 = math.atan2(sin_theta1, cos_theta1)

    def rotation_z(theta):
        return np.array([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

    def rotation_y(theta):
        return np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
    R1 = rotation_z(theta1)
    R2 = rotation_y(theta2)
    joint2_pos = np.array([0, 0.13585, 0])
    P1 = R1 @ joint2_pos
    joint3_pos = np.array([0, -0.1197, 0.425])
    P2 = R1 @ R2 @ joint3_pos
    P12 = P1 + P2
    v = np.array(p) - P12
    R12 = R1 @ R2
    v_prime = R12.T @ v
    theta3 = math.atan2(v_prime[0], v_prime[2])
    return (theta1, theta2, theta3)