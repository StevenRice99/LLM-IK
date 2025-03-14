import math
import numpy as np
from sympy import symbols, Eq, solve, sin, cos

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    rx, ry, rz = r
    θ1, θ2 = symbols('θ1 θ2')
    A = 0.093
    B = 0.09465
    x_fk = -A * sin(θ2) * cos(θ1) + B * cos(θ1)
    y_fk = A * cos(θ2) * cos(θ1) + B * sin(θ1)
    z_fk = A * sin(θ2) * sin(θ1) + B * cos(θ1)
    R1 = [[cos(θ1), sin(θ1), 0], [0, cos(θ1), -sin(θ1)], [-sin(θ1), 0, cos(θ1)]]
    R2 = [[cos(θ2), -sin(θ2), 0], [sin(θ2), cos(θ2), 0], [0, 0, 1]]
    R_total = np.dot(R2, R1)
    R_desired = [[cos(rz) * cos(ry) - sin(rz) * sin(rx) * sin(ry), cos(rz) * sin(ry) * sin(rx) + sin(rz) * cos(rx), cos(rz) * sin(ry) * cos(rx) - sin(rz) * sin(rx)], [sin(rz) * cos(ry) + cos(rz) * sin(rx) * sin(ry), sin(rz) * sin(ry) * sin(rx) - cos(rz) * cos(rx), sin(rz) * sin(ry) * cos(rx) + cos(rz) * sin(rx)], [-sin(ry), sin(rx) * cos(ry), cos(rx) * cos(ry)]]
    eq1 = Eq(x_fk, x)
    eq2 = Eq(y_fk, y)
    eq3 = Eq(z_fk, z)
    eq4 = Eq(R_total[0][0], R_desired[0][0])
    eq5 = Eq(R_total[0][1], R_desired[0][1])
    eq6 = Eq(R_total[0][2], R_desired[0][2])
    solution = solve((eq1, eq2, eq3, eq4, eq5, eq6), (θ1, θ2))
    θ1_val = solution[θ1]
    θ2_val = solution[θ2]
    return (θ1_val, θ2_val)