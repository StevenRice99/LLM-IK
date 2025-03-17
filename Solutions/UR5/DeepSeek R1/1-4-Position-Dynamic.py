import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x_target, y_target, z_target = p
    a = 0.425
    b = 0.39225
    c = 0.18765
    phi = math.atan2(x_target, z_target)
    x_prime = x_target - c * math.sin(phi)
    z_prime = z_target - c * math.cos(phi)
    dx = x_prime
    dz = z_prime
    d_sq = dx ** 2 + dz ** 2
    cos_theta2 = (d_sq - a ** 2 - b ** 2) / (2 * a * b)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2_positive = math.acos(cos_theta2)
    theta2_negative = -theta2_positive
    gamma = math.atan2(dx, dz)
    delta_up = math.atan2(b * math.sin(theta2_positive), a + b * math.cos(theta2_positive))
    theta1_up = gamma - delta_up
    theta3_up = phi - (theta1_up + theta2_positive)
    delta_down = math.atan2(b * math.sin(theta2_negative), a + b * math.cos(theta2_negative))
    theta1_down = gamma - delta_down
    theta3_down = phi - (theta1_down + theta2_negative)
    if abs(theta3_up) <= abs(theta3_down):
        theta1, theta2, theta3 = (theta1_up, theta2_positive, theta3_up)
    else:
        theta1, theta2, theta3 = (theta1_down, theta2_negative, theta3_down)
    theta4 = 0.0
    return (theta1, theta2, theta3, theta4)