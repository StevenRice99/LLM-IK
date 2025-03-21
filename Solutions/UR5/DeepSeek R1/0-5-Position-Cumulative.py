import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    x_tcp, y_tcp, z_tcp = p
    a = 0.0823
    L1 = 0.093
    L2 = 0.09465
    x_wc = x_tcp + 0.1753
    y_wc = y_tcp
    z_wc = z_tcp - L2
    k = 0.01615
    k_sq = k ** 2
    x, y, z = (x_wc, y_wc, z_wc)
    numerator_cosθ3 = x ** 2 + y ** 2 + z ** 2 - 0.334746
    denominator_cosθ3 = 0.3334125
    cosθ3 = numerator_cosθ3 / denominator_cosθ3
    cosθ3 = max(min(cosθ3, 1.0), -1.0)
    θ3_options = []
    if abs(cosθ3) <= 1.0:
        θ3_pos = math.acos(cosθ3)
        θ3_neg = -θ3_pos
        θ3_options = [θ3_pos, θ3_neg]
    valid_solutions = []
    for θ3 in θ3_options:
        C = 0.425 + 0.39225 * math.cos(θ3)
        D = 0.39225 * math.sin(θ3)
        A_sq = x ** 2 + y ** 2 - k_sq
        if A_sq < 0:
            continue
        A = math.sqrt(A_sq)
        denominator_theta2 = C ** 2 + D ** 2
        if denominator_theta2 < 1e-06:
            continue
        sinθ2 = (C * A - D * z) / denominator_theta2
        cosθ2 = (C * z + D * A) / denominator_theta2
        θ2 = math.atan2(sinθ2, cosθ2)
        denominator_theta1 = x ** 2 + y ** 2
        if denominator_theta1 == 0:
            θ1 = 0.0
        else:
            numerator_cosθ1 = A * x + k * y
            numerator_sinθ1 = A * y - k * x
            θ1 = math.atan2(numerator_sinθ1, numerator_cosθ1)
        valid_solutions.append((θ1, θ2, θ3))
    if not valid_solutions:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    θ1, θ2, θ3 = valid_solutions[0]
    R_z = np.array([[math.cos(θ1), -math.sin(θ1), 0], [math.sin(θ1), math.cos(θ1), 0], [0, 0, 1]])
    R_y_2_3 = np.array([[math.cos(θ2 + θ3), 0, math.sin(θ2 + θ3)], [0, 1, 0], [-math.sin(θ2 + θ3), 0, math.cos(θ2 + θ3)]])
    R_03 = R_z @ R_y_2_3
    R_desired = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R_03_T = R_03.T
    R_36 = R_03_T @ R_desired
    theta5 = math.acos(R_36[2, 2])
    theta5 = math.copysign(theta5, R_36[2, 2])
    valid_wrist = []
    for t5 in [theta5, -theta5]:
        if abs(t5) < 1e-06:
            theta4 = 0.0
            theta6 = math.atan2(R_36[0, 1], R_36[0, 0])
        else:
            sin_t5 = math.sin(t5)
            if abs(sin_t5) < 1e-06:
                continue
            theta4 = math.atan2(R_36[0, 2] / sin_t5, R_36[1, 2] / sin_t5)
            theta6 = math.atan2(R_36[2, 0] / sin_t5, -R_36[2, 1] / sin_t5)
        valid_wrist.append((theta4, t5, theta6))
    if valid_wrist:
        theta4, theta5, theta6 = valid_wrist[0]
    else:
        theta4, theta5, theta6 = (0.0, 0.0, 0.0)
    return (θ1, θ2, θ3, theta4, theta5, theta6)