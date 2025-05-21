import math
import numpy as np

def rotz(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def roty(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotx(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    px, py, pz = p
    roll_x, pitch_y, yaw_z = r
    L1y = 0.13585
    L2y = -0.1197
    L2z = 0.425
    L3z = 0.39225
    L4y = 0.093
    L5z = 0.09465
    L6y_tcp = 0.0823
    tcp_rotz_val = 1.570796325
    P_target_F0 = np.array([px, py, pz])
    R_target_F0 = rotz(yaw_z) @ roty(pitch_y) @ rotx(roll_x)
    R_F6_TCP = rotz(tcp_rotz_val)
    P_TCP_in_F6 = np.array([0, L6y_tcp, 0])
    R_F0_F6 = R_target_F0 @ R_F6_TCP.T
    P_O6_in_F0 = P_target_F0 - R_F0_F6 @ P_TCP_in_F6
    q1, q2, q3, q4, q5, q6 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return (q1, q2, q3, q4, q5, q6)