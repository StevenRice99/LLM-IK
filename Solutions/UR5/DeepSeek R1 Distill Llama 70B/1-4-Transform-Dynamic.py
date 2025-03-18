import math
from sympy import symbols, Eq, solve

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Solves for the joint angles required to reach the desired position and orientation.
    
    Args:
        p: Desired position (x, y, z)
        r: Desired orientation (roll, pitch, yaw in radians)
        
    Returns:
        Tuple of joint angles (θ1, θ2, θ3, θ4)
    """
    θ1, θ2, θ3, θ4 = symbols('θ1 θ2 θ3 θ4')
    eq1 = Eq(0, p[0])
    eq2 = Eq(-0.0267, p[1])
    eq3 = Eq(0.9119, p[2])
    eq4 = Eq(0, r[0])
    eq5 = Eq(0, r[1])
    eq6 = Eq(0, r[2])
    solution = solve((eq1, eq2, eq3, eq4, eq5, eq6), (θ1, θ2, θ3, θ4))
    q1 = float(solution[θ1])
    q2 = float(solution[θ2])
    q3 = float(solution[θ3])
    q4 = float(solution[θ4])
    return (q1, q2, q3, q4)