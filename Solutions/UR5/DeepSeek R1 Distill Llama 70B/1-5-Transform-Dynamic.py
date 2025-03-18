import math
from typing import Tuple
import sympy as sp

def inverse_kinematics(p: Tuple[float, float, float], r: Tuple[float, float, float]) -> Tuple[float, float, float, float, float]:
    """
    Solves for the joint angles to reach the desired position and orientation using inverse kinematics.
    
    Args:
        p: The desired position (x, y, z).
        r: The desired orientation as Euler angles (rx, ry, rz).
    
    Returns:
        A tuple of five joint angles in radians.
    """
    theta1, theta2, theta3, theta4, theta5 = sp.symbols('theta1 theta2 theta3 theta4 theta5')
    dh_params = [{'alpha': 0, 'a': 0, 'd': 0, 'theta': theta1}, {'alpha': 0, 'a': -0.1197, 'd': 0.425, 'theta': theta2}, {'alpha': 0, 'a': 0, 'd': 0.39225, 'theta': theta3}, {'alpha': sp.pi / 2, 'a': 0.093, 'd': 0, 'theta': theta4}, {'alpha': -sp.pi / 2, 'a': 0, 'd': 0.09465, 'theta': theta5}]

    def transformation_matrix(params):
        alpha = params['alpha']
        a = params['a']
        d = params['d']
        theta = params['theta']
        return sp.Matrix([[sp.cos(theta), -sp.sin(theta) * sp.cos(alpha), sp.sin(theta) * sp.sin(alpha), a * sp.cos(theta)], [sp.sin(theta), sp.cos(theta) * sp.cos(alpha), -sp.cos(theta) * sp.sin(alpha), a * sp.sin(theta)], [0, sp.sin(alpha), sp.cos(alpha), d], [0, 0, 0, 1]])
    T1 = transformation_matrix(dh_params[0])
    T2 = transformation_matrix(dh_params[1])
    T3 = transformation_matrix(dh_params[2])
    T4 = transformation_matrix(dh_params[3])
    T5 = transformation_matrix(dh_params[4])
    T_tcp = T1 * T2 * T3 * T4 * T5
    x_tcp = T_tcp[0, 3]
    y_tcp = T_tcp[1, 3]
    z_tcp = T_tcp[2, 3]
    rx = sp.atan2(T_tcp[1, 2], T_tcp[0, 2])
    ry = sp.atan2(T_tcp[2, 1], T_tcp[2, 2])
    rz = sp.atan2(T_tcp[1, 0], T_tcp[0, 0])
    equations = [sp.Eq(x_tcp, p[0]), sp.Eq(y_tcp, p[1]), sp.Eq(z_tcp, p[2]), sp.Eq(rx, r[0]), sp.Eq(ry, r[1]), sp.Eq(rz, r[2])]
    solution = sp.solve(equations, (theta1, theta2, theta3, theta4, theta5))
    return tuple((float(angle) for angle in solution.values()))