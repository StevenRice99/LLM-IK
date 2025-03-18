import math
from sympy import symbols, Eq, solve

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    q1, q2, q3, q4 = symbols('q1 q2 q3 q4')
    eq1 = Eq(x, q1 + q2)
    eq2 = Eq(y, q3 + q4)
    eq3 = Eq(z, q2 + q3)
    solution = solve((eq1, eq2, eq3), (q1, q2, q3, q4))
    q1_val = solution[q1]
    q2_val = solution[q2]
    q3_val = solution[q3]
    q4_val = solution[q4]
    return (float(q1_val), float(q2_val), float(q3_val), float(q4_val))