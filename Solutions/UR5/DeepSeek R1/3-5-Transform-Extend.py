def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    import numpy as np
    import math
    x, y, z = p
    rx, ry, rz = r
    y_offset = y - 0.093
    cos_theta2 = y_offset / 0.0823
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2_positive = math.acos(cos_theta2)
    theta2_candidates = [theta2_positive, -theta2_positive]

    def rot_x(angle):
        return np.array([[1, 0, 0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]])

    def rot_y(angle):
        return np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])

    def rot_z(angle):
        return np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    for theta2 in theta2_candidates:
        A = -0.0823 * math.sin(theta2)
        B = 0.09465
        denominator = A ** 2 - B ** 2
        if abs(denominator) < 1e-06:
            continue
        sin_theta1 = (A * z - B * x) / denominator
        cos_theta1 = (A * x - B * z) / denominator
        norm = math.hypot(sin_theta1, cos_theta1)
        if abs(norm) < 1e-06:
            continue
        sin_theta1 /= norm
        cos_theta1 /= norm
        theta1 = math.atan2(sin_theta1, cos_theta1)
        R_target = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
        R1 = rot_y(theta1)
        R2 = rot_z(theta2)
        Rz_pi2 = rot_z(math.pi / 2)
        R_remaining = R2.T @ R1.T @ R_target @ Rz_pi2.T
        theta3 = math.atan2(R_remaining[0, 2], R_remaining[0, 0])
        return (theta1, theta2, theta3)
    theta2 = theta2_candidates[0]
    A = -0.0823 * math.sin(theta2)
    B = 0.09465
    denominator = A ** 2 - B ** 2
    sin_theta1 = (A * z - B * x) / denominator
    cos_theta1 = (A * x - B * z) / denominator
    theta1 = math.atan2(sin_theta1, cos_theta1)
    R_target = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    R1 = rot_y(theta1)
    R2 = rot_z(theta2)
    Rz_pi2 = rot_z(math.pi / 2)
    R_remaining = R2.T @ R1.T @ R_target @ Rz_pi2.T
    theta3 = math.atan2(R_remaining[0, 2], R_remaining[0, 0])
    return (theta1, theta2, theta3)