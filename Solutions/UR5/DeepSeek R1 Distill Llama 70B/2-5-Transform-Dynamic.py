import math
from sympy import symbols, Eq, solve

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Solves the inverse kinematics problem for a 4-DOF serial manipulator.
    
    Args:
        p: Target position (x, y, z)
        r: Target orientation (roll, pitch, yaw in radians)
    
    Returns:
        Tuple of joint angles (joint1, joint2, joint3, joint4) in radians.
    """
    q1, q2, q3, q4 = symbols('q1 q2 q3 q4')
    x = q1 + q2 * 0.1
    y = q3 * 0.2
    z = q4 * 0.3
    roll = q1
    pitch = q2
    yaw = q3 + q4
    eq1 = Eq(x, p[0])
    eq2 = Eq(y, p[1])
    eq3 = Eq(z, p[2])
    eq4 = Eq(roll, r[0])
    eq5 = Eq(pitch, r[1])
    eq6 = Eq(yaw, r[2])
    solutions = solve((eq1, eq2, eq3, eq4, eq5, eq6), (q1, q2, q3, q4))
    if solutions:
        solution = solutions[0]
        q1_val = float(solution[q1])
        q2_val = float(solution[q2])
        q3_val = float(solution[q3])
        q4_val = float(solution[q4])
        if all((-6.2831853 <= q <= 6.2831853 for q in [q1_val, q2_val, q3_val, q4_val])):
            return (q1_val, q2_val, q3_val, q4_val)
        else:
            return (0.0, 0.0, 0.0, 0.0)
    else:
        return (0.0, 0.0, 0.0, 0.0)