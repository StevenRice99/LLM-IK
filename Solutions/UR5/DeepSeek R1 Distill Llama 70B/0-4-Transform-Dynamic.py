import math
from sympy import symbols, Eq, solve, Matrix, sin, cos

def transformation_matrix(angle, axis, translation):
    """
    Returns a 4x4 homogeneous transformation matrix for the given angle, axis, and translation.
    """
    rot = rotation_matrix(angle, axis)
    trans = Matrix([[0, 0, 0, translation[0]], [0, 0, 0, translation[1]], [0, 0, 0, translation[2]], [0, 0, 0, 1]])
    return Matrix([[rot[0, 0], rot[0, 1], rot[0, 2], translation[0]], [rot[1, 0], rot[1, 1], rot[1, 2], translation[1]], [rot[2, 0], rot[2, 1], rot[2, 2], translation[2]], [0, 0, 0, 1]])

def rotation_matrix(angle, axis):
    """
    Returns a 3x3 rotation matrix for the given angle and axis.
    """
    if axis == 'x':
        return Matrix([[1, 0, 0], [0, cos(angle), -sin(angle)], [0, sin(angle), cos(angle)]])
    elif axis == 'y':
        return Matrix([[cos(angle), 0, sin(angle)], [0, 1, 0], [-sin(angle), 0, cos(angle)]])
    elif axis == 'z':
        return Matrix([[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]])
    else:
        raise ValueError('Invalid rotation axis')

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [roll, pitch, yaw].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    q1, q2, q3, q4, q5 = symbols('q1 q2 q3 q4 q5')
    T1 = transformation_matrix(q1, 'z', [0, 0, 0])
    T2 = transformation_matrix(q2, 'y', [0, 0.13585, 0])
    T3 = transformation_matrix(q3, 'y', [0, -0.1197, 0.425])
    T4 = transformation_matrix(q4, 'y', [0, 0, 0.39225])
    T5 = transformation_matrix(q5, 'z', [0, 0.093, 0])
    T_total = T1 * T2 * T3 * T4 * T5
    x = T_total[0, 3]
    y = T_total[1, 3]
    z = T_total[2, 3]
    roll = T_total[0, 0]
    pitch = T_total[1, 0]
    yaw = T_total[2, 0]
    equations = [Eq(x, p[0]), Eq(y, p[1]), Eq(z, p[2]), Eq(roll, r[0]), Eq(pitch, r[1]), Eq(yaw, r[2])]
    solution = solve(equations, (q1, q2, q3, q4, q5))
    return tuple(solution.values())