import math
import numpy as np

def euler_to_matrix(roll, pitch, yaw):
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    return Rz @ Ry @ Rx

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    px, py, pz = p
    roll, pitch, yaw = r
    R_target = euler_to_matrix(roll, pitch, yaw)
    theta2 = math.atan2(-R_target[2, 0], R_target[2, 2])
    theta2_alt = theta2 + math.pi
    for theta2_candidate in [theta2, theta2_alt]:
        Ry_theta2 = np.array([[math.cos(theta2_candidate), 0, math.sin(theta2_candidate)], [0, 1, 0], [-math.sin(theta2_candidate), 0, math.cos(theta2_candidate)]])
        Rz_theta1 = R_target @ np.linalg.inv(Ry_theta2)
        theta1 = math.atan2(Rz_theta1[1, 0], Rz_theta1[0, 0])
        theta1_alt = theta1 + math.pi
        for theta1_candidate in [theta1, theta1_alt]:
            sin_theta1 = math.sin(theta1_candidate)
            cos_theta1 = math.cos(theta1_candidate)
            sin_theta2 = math.sin(theta2_candidate)
            cos_theta2 = math.cos(theta2_candidate)
            x_calculated = 0.425 * sin_theta2 * cos_theta1 - 0.01615 * sin_theta1
            y_calculated = 0.425 * sin_theta2 * sin_theta1 + 0.01615 * cos_theta1
            z_calculated = 0.425 * cos_theta2
            if math.isclose(x_calculated, px, abs_tol=1e-06) and math.isclose(y_calculated, py, abs_tol=1e-06) and math.isclose(z_calculated, pz, abs_tol=1e-06):
                theta1_norm = (theta1_candidate + math.pi) % (2 * math.pi) - math.pi
                theta2_norm = (theta2_candidate + math.pi) % (2 * math.pi) - math.pi
                return (theta1_norm, theta2_norm)
    return (theta1, theta2)