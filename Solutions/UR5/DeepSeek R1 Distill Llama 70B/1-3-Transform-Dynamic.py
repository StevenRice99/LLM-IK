import math
from sympy import symbols, Eq, solve

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    rx, ry, rz = r
    theta1, theta2, theta3 = symbols('theta1 theta2 theta3')
    eq1 = Eq(x, theta1 + theta2)
    eq2 = Eq(y, theta2 + theta3)
    eq3 = Eq(z, theta3)
    solution = solve((eq1, eq2, eq3), (theta1, theta2, theta3))
    return (solution[theta1], solution[theta2], solution[theta3])