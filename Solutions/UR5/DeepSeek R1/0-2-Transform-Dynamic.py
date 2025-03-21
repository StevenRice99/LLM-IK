import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    x_target, y_target, z_target = p
    rx, ry, rz = r
    A = y_target
    B = -x_target
    C = 0.13585
    D = math.hypot(A, B)
    if D < 1e-09:
        theta1 = 0.0
    else:
        phi = math.atan2(B, A)
        acos_arg = max(min(C / D, 1.0), -1.0)
        delta = math.acos(acos_arg)
        theta1_1 = phi + delta
        theta1_2 = phi - delta
        if abs(theta1_1 - rz) <= abs(theta1_2 - rz):
            theta1 = theta1_1
        else:
            theta1 = theta1_2
    x_revolute2 = x_target * math.cos(theta1) + y_target * math.sin(theta1)
    z_revolute2 = z_target
    a = math.hypot(-0.1197, 0.425)
    b = 0.39225
    theta_sum = ry
    term_x = x_revolute2 - b * math.sin(theta_sum)
    term_z = z_revolute2 - b * math.cos(theta_sum)
    theta2 = math.atan2(term_x, term_z)
    theta3 = theta_sum - theta2
    return (theta1, theta2, theta3)