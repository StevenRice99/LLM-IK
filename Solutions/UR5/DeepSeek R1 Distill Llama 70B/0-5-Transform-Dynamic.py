import numpy as np
from sympy import symbols, sin, cos, atan2, Eq, solve

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Solves the inverse kinematics problem for a 6-DOF robot arm using symbolic computation.
    
    Args:
        p: The target position (x, y, z) of the TCP.
        r: The target orientation (rx, ry, rz) of the TCP in radians.
    
    Returns:
        A tuple of six joint angles (θ1, θ2, θ3, θ4, θ5, θ6) in radians.
    """
    θ1, θ2, θ3, θ4, θ5, θ6 = symbols('θ1 θ2 θ3 θ4 θ5 θ6')

    def transformation_matrix(theta, d, a, alpha):
        return np.array([[cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)], [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)], [0, sin(alpha), cos(alpha), d], [0, 0, 0, 1]], dtype=object)
    T = np.eye(4, dtype=object)
    for i in range(6):
        if i == 0:
            a = 0
            d = 0
            alpha = 0
        elif i == 1:
            a = 0.13585
            d = 0
            alpha = np.pi / 2
        elif i == 2:
            a = 0.425
            d = 0.1197
            alpha = -np.pi / 2
        elif i == 3:
            a = 0.39225
            d = 0
            alpha = np.pi / 2
        elif i == 4:
            a = 0.093
            d = 0
            alpha = -np.pi / 2
        elif i == 5:
            a = 0.09465
            d = 0.0823
            alpha = np.pi / 2
        theta = [θ1, θ2 - np.pi / 2, θ3, θ4, θ5, θ6][i]
        T_link = transformation_matrix(theta, d, a, alpha)
        T = np.dot(T, T_link)
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]
    rx = atan2(T[2, 1], T[2, 2])
    ry = atan2(-T[2, 0], (T[2, 1] ** 2 + T[2, 2] ** 2) ** 0.5)
    rz = atan2(T[1, 0], T[0, 0])
    px, py, pz = p
    rrx, rry, rrz = r
    eq1 = Eq(x, px)
    eq2 = Eq(y, py)
    eq3 = Eq(z, pz)
    eq4 = Eq(rx, rrx)
    eq5 = Eq(ry, rry)
    eq6 = Eq(rz, rrz)
    solution = solve((eq1, eq2, eq3, eq4, eq5, eq6), (θ1, θ2, θ3, θ4, θ5, θ6))
    θ1_val = solution[θ1]
    θ2_val = solution[θ2]
    θ3_val = solution[θ3]
    θ4_val = solution[θ4]
    θ5_val = solution[θ5]
    θ6_val = solution[θ6]
    return (θ1_val, θ2_val, θ3_val, θ4_val, θ5_val, θ6_val)