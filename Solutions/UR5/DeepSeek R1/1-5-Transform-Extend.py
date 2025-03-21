import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    x_target, y_target, z_target = p
    roll, pitch, yaw = r
    cr, sr = (math.cos(roll), math.sin(roll))
    cp, sp = (math.cos(pitch), math.sin(pitch))
    cy, sy = (math.cos(yaw), math.sin(yaw))
    R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R_target = R_z @ R_y @ R_x
    wrist_y = -0.1197 + 0.093
    K = (y_target - wrist_y) / 0.0823
    r10 = R_target[1, 0]
    r11 = R_target[1, 1]
    r12 = R_target[1, 2]
    theta5 = math.atan2(r12, r10) if not (np.isclose(r10, 0) and np.isclose(r12, 0)) else 0.0
    sin_theta4 = math.sqrt(r10 ** 2 + r12 ** 2)
    theta4 = math.atan2(sin_theta4, r11)
    cz, sz = (math.cos(theta4), math.sin(theta4))
    cy5, sy5 = (math.cos(theta5), math.sin(theta5))
    offset_x = 0.0823 * sy5 * cz
    offset_z = 0.09465 + 0.0823 * cy5
    x_wrist = x_target - offset_x
    z_wrist = z_target - offset_z
    a, b = (0.425, 0.39225)
    x, z = (x_wrist, z_wrist)
    d_sq = x ** 2 + z ** 2
    cos_theta2 = (d_sq - a ** 2 - b ** 2) / (2 * a * b)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2_1 = math.acos(cos_theta2)
    theta2_2 = -theta2_1
    solutions = []
    for theta2 in [theta2_1, theta2_2]:
        denom = a + b * math.cos(theta2)
        num = b * math.sin(theta2)
        theta1 = math.atan2(x, z) - math.atan2(num, denom)
        x_calc = a * math.sin(theta1) + b * math.sin(theta1 + theta2)
        z_calc = a * math.cos(theta1) + b * math.cos(theta1 + theta2)
        if math.isclose(x_calc, x, abs_tol=1e-05) and math.isclose(z_calc, z, abs_tol=1e-05):
            theta_sum = math.atan2(R_target[0, 2], R_target[2, 2])
            theta3 = theta_sum - theta1 - theta2
            solutions.append((theta1, theta2, theta3, theta4, theta5))
    if not solutions:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        return solutions[0]