import math
from sympy import symbols, Eq, solve, sin, cos, Matrix, eye

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    q1, q2, q3, q4, q5 = symbols('q1 q2 q3 q4 q5')
    dh_parameters = [{'a': 0, 'alpha': 0, 'd': 0, 'theta': q1}, {'a': 0.13585, 'alpha': -math.pi / 2, 'd': 0.1197, 'theta': q2}, {'a': -0.1197, 'alpha': math.pi / 2, 'd': 0.425, 'theta': q3}, {'a': 0, 'alpha': -math.pi / 2, 'd': 0.39225, 'theta': q4}, {'a': 0.093, 'alpha': math.pi / 2, 'd': 0.09465, 'theta': q5}]

    def dh_matrix(a, alpha, d, theta):
        st = sin(theta)
        ct = cos(theta)
        sa = sin(alpha)
        ca = cos(alpha)
        rotation = Matrix([[ct, -st * ca, st * sa], [st, ct * ca, -ct * sa], [0, sa, ca]])
        translation = Matrix([[a, 0, d], [0, a, 0], [0, 0, a]])
        dh = eye(4)
        dh[:3, :3] = rotation
        dh[:3, 3] = translation[:3, 0]
        return dh
    transformations = []
    for params in dh_parameters:
        a = params['a']
        alpha = params['alpha']
        d = params['d']
        theta = params['theta']
        transformations.append(dh_matrix(a, alpha, d, theta))
    overall_transform = eye(4)
    for transform in transformations:
        overall_transform = overall_transform * transform
    tx, ty, tz = (overall_transform[0, 3], overall_transform[1, 3], overall_transform[2, 3])
    rx = overall_transform[0, 0]
    ry = overall_transform[1, 1]
    rz = overall_transform[2, 2]
    equations = [Eq(tx, p[0]), Eq(ty, p[1]), Eq(tz, p[2]), Eq(rx, r[0]), Eq(ry, r[1]), Eq(rz, r[2])]
    solution = solve(equations, (q1, q2, q3, q4, q5))
    return (float(solution[q1]), float(solution[q2]), float(solution[q3]), float(solution[q4]), float(solution[q5]))