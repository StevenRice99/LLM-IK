import math
from sympy import symbols, cos, sin, atan2, acos, sqrt, Matrix
from sympy import Euler2RotationMatrix as e2rm

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    rx, ry, rz = r
    theta1, theta2 = symbols('theta1 theta2')
    Rz = Matrix([[cos(theta1), -sin(theta1), 0], [sin(theta1), cos(theta1), 0], [0, 0, 1]])
    Ry = Matrix([[cos(theta2), 0, sin(theta2)], [0, 1, 0], [-sin(theta2), 0, cos(theta2)]])
    R_total = Rz * Ry
    R_desired = Matrix(e2rm(rx, ry, rz, 'rzyx'))
    equations = []
    for i in range(3):
        for j in range(3):
            equations.append(R_total[i, j] - R_desired[i, j])
    solution = {}
    theta2_sol = acos(z / 0.425)
    sin_theta2 = sin(theta2_sol)
    cos_theta2 = cos(theta2_sol)
    K = 0.425 * sin_theta2
    L = 0.01615
    denominator = K ** 2 + L ** 2
    cos_theta1 = (K * x + L * y) / denominator
    sin_theta1 = (-L * x + K * y) / denominator
    theta1_sol = atan2(sin_theta1, cos_theta1)
    return (float(theta1_sol), float(theta2_sol))