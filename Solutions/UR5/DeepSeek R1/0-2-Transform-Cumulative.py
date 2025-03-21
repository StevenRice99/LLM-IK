import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    x_target, y_target, z_target = p
    rx, ry, rz = r
    cx = math.cos(rx)
    sx = math.sin(rx)
    cy = math.cos(ry)
    sy = math.sin(ry)
    cz = math.cos(rz)
    sz = math.sin(rz)
    R_target = [[cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx], [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx], [-sy, cy * sx, cy * cx]]
    theta1 = math.atan2(R_target[1][0], R_target[0][0])
    theta_sum = math.atan2(-R_target[2][0], R_target[2][2])
    if abs(math.cos(theta1)) > 1e-09:
        S = (x_target - 0.25555 * math.sin(theta1)) / math.cos(theta1)
    else:
        S = (y_target - 0.01615 * math.cos(theta1)) / math.sin(theta1)
    sin_theta2 = (S - 0.39225 * math.sin(theta_sum)) / 0.425
    cos_theta2 = (z_target - 0.39225 * math.cos(theta_sum)) / 0.425
    norm = math.hypot(sin_theta2, cos_theta2)
    sin_theta2 /= norm
    cos_theta2 /= norm
    theta2 = math.atan2(sin_theta2, cos_theta2)
    theta3 = theta_sum - theta2
    return (theta1, theta2, theta3)