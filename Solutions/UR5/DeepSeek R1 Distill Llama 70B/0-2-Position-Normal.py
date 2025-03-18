import numpy as np
import sympy as sp

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    q1, q2, q3 = sp.symbols('q1 q2 q3')
    T0_1 = sp.Matrix([[sp.cos(q1), -sp.sin(q1), 0, 0], [sp.sin(q1), sp.cos(q1), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T1_2 = sp.Matrix([[sp.cos(q2), 0, sp.sin(q2), 0.13585 * sp.cos(q2)], [sp.sin(q2), 0, -sp.cos(q2), 0.13585 * sp.sin(q2)], [0, 1, 0, 0], [0, 0, 0, 1]])
    T2_3 = sp.Matrix([[sp.cos(q3), 0, sp.sin(q3), -0.1197 * sp.cos(q3)], [sp.sin(q3), 0, -sp.cos(q3), -0.1197 * sp.sin(q3)], [0, 1, 0, 0.425], [0, 0, 0, 1]])
    T3_TCP = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.39225], [0, 0, 0, 1]])
    T0_TCP = T0_1 * T1_2 * T2_3 * T3_TCP
    x = T0_TCP[0, 3]
    y = T0_TCP[1, 3]
    z = T0_TCP[2, 3]
    eq1 = sp.Eq(x, p[0])
    eq2 = sp.Eq(y, p[1])
    eq3 = sp.Eq(z, p[2])
    solution = sp.solve((eq1, eq2, eq3), (q1, q2, q3))
    return (float(solution[q1]), float(solution[q2]), float(solution[q3]))