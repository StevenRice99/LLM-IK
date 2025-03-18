import numpy as np
from sympy import symbols, Matrix, cos, sin, atan2, sqrt

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    r_x, r_y, r_z = r
    L1 = 0.093
    L2 = 0.09465
    L3 = 0.0823
    theta1, theta2, theta3 = symbols('theta1 theta2 theta3')
    T1 = Matrix([[cos(theta1), 0, sin(theta1), 0], [0, 1, 0, 0], [-sin(theta1), 0, cos(theta1), 0], [0, 0, 0, 1]])
    T2 = Matrix([[cos(theta2), -sin(theta2), 0, 0], [sin(theta2), cos(theta2), 0, L1], [0, 0, 1, 0], [0, 0, 0, 1]])
    T3 = Matrix([[cos(theta3), 0, sin(theta3), 0], [0, 1, 0, 0], [-sin(theta3), 0, cos(theta3), L2], [0, 0, 0, 1]])
    T_tcp = Matrix([[1, 0, 0, 0], [0, 1, 0, L3], [0, 0, 1, 0], [0, 0, 0, 1]])
    T = T1 * T2 * T3 * T_tcp
    px, py, pz = (T[0, 3], T[1, 3], T[2, 3])
    r11, r12, r13 = (T[0, 0], T[0, 1], T[0, 2])
    r21, r22, r23 = (T[1, 0], T[1, 1], T[1, 2])
    r31, r32, r33 = (T[2, 0], T[2, 1], T[2, 2])
    theta1_sol = atan2(x, z)
    y_prime = y - L3
    theta2_sol = atan2(y_prime, sqrt(x ** 2 + z ** 2))
    z_prime = z - L1
    theta3_sol = atan2(z_prime, L2)
    return (float(theta1_sol), float(theta2_sol), float(theta3_sol))