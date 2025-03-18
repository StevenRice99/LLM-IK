import math
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
    eq1 = Eq(-0.0823 * sin(θ1) * sin(θ2), x)
    eq2 = Eq(0.0823 * cos(θ1) * sin(θ2) + 0.09465 * cos(θ2), y)
    eq3 = Eq(0.0823 * sin(θ1) * cos(θ2) + 0.09465 * sin(θ2), z)
    solution = solve((eq1, eq2, eq3), (θ1, θ2))
    return (float(solution[θ1]), float(solution[θ2]))