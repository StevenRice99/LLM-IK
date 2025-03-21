import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x_target, y_target, z_target = p
    L1 = 0.39225
    L2 = 0.093
    L3 = 0.09465
    a = 0.0823
    y_eff = y_target - L2
    if abs(y_eff) > a:
        raise ValueError('Target Y position unreachable')
    theta4 = math.asin(y_eff / a)
    theta3 = 0.0
    eff_x = x_target - a * math.cos(theta4) * math.sin(theta3)
    eff_z = z_target - (L3 + a * math.cos(theta4) * math.cos(theta3))
    theta1 = math.atan2(eff_x, eff_z)
    adjusted_x = eff_x - L1 * math.sin(theta1)
    adjusted_z = eff_z - L1 * math.cos(theta1)
    theta_total = math.atan2(adjusted_x, adjusted_z)
    theta2 = theta_total - theta1
    return (theta1, theta2, theta3, theta4)