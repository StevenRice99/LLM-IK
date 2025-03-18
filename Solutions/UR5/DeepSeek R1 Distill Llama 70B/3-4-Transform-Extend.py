import math
import sympy as sp

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    theta1, theta2 = sp.symbols('theta1 theta2')
    x = 0.093 * sp.cos(theta1 + theta2)
    y = 0.093 * sp.sin(theta1 + theta2)
    z = 0.09465
    eq_pos = [sp.Eq(x, p[0]), sp.Eq(y, p[1]), sp.Eq(z, p[2])]
    solution = sp.solve(eq_pos, (theta1, theta2))
    if not solution:
        raise ValueError('No solution found.')
    first_solution = solution[0]
    theta1_val = first_solution[theta1]
    theta2_val = first_solution[theta2]
    return (float(theta1_val), float(theta2_val))