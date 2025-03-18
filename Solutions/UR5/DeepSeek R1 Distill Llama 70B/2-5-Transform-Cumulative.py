import math
from sympy import symbols, Eq, solve, cos, sin

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    q1, q2, q3, q4 = symbols('q1 q2 q3 q4')
    T1 = rotation_matrix(q1, 'Y')
    T2 = rotation_matrix(q2, 'Y')
    T3 = rotation_matrix(q3, 'Z')
    T4 = rotation_matrix(q4, 'Y')
    T = T1 * T2 * T3 * T4
    x, y, z = T[0:3, 3]
    rx, ry, rz = euler_from_rotation_matrix(T[0:3, 0:3])
    eqs = [Eq(x, p[0]), Eq(y, p[1]), Eq(z, p[2]), Eq(rx, r[0]), Eq(ry, r[1]), Eq(rz, r[2])]
    solution = solve(eqs, (q1, q2, q3, q4))
    return (solution[q1], solution[q2], solution[q3], solution[q4])

def rotation_matrix(angle, axis):
    if axis == 'X':
        return [[1, 0, 0], [0, cos(angle), -sin(angle)], [0, sin(angle), cos(angle)]]
    elif axis == 'Y':
        return [[cos(angle), 0, sin(angle)], [0, 1, 0], [-sin(angle), 0, cos(angle)]]
    elif axis == 'Z':
        return [[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]]
    else:
        raise ValueError('Invalid rotation axis')

def euler_from_rotation_matrix(R):
    sy = (R[0, 0] ** 2 + R[1, 0] ** 2) ** 0.5
    singular = sy < 1e-06
    if not singular:
        x = atan2(R[2, 1], R[2, 2])
        y = atan2(R[1, 2], R[0, 2])
        z = atan2(R[2, 0], R[2, 1])
    else:
        x = atan2(-R[1, 2], R[1, 1])
        y = atan2(-R[2, 2], R[0, 2])
        z = 0
    return (x, y, z)