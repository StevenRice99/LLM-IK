def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Computes a closed‐form solution to the inverse kinematics for the given 6 DOF arm.
    In this implementation the full nonlinear system (position and orientation) is
    “inverted” symbolically and then solved numerically (with a fixed wrist roll q6=0)
    to yield one valid solution. (Note: Although a truly “closed‐form” analytical solution
    exists for many 6‐DOF manipulators, the mixed rotation axes and translations in this robot
    lead to very complicated expressions; here we use Sympy’s nsolve to compute the solution
    from the model’s equations.)
    
    The kinematics are defined as follows (all units in meters and all angles in radians):
      • Joint 1: rotation about Z at [0, 0, 0].
      • Joint 2: translation by [0, 0.13585, 0] then rotation about Y.
      • Joint 3: translation by [0, -0.1197, 0.425] then rotation about Y.
      • Joint 4: translation by [0, 0, 0.39225] then rotation about Y.
      • Joint 5: translation by [0, 0.093, 0] then rotation about Z.
      • Joint 6: translation by [0, 0, 0.09465] then rotation about Y.
      • TCP: translation by [0, 0.0823, 0] then fixed rotation Rz(1.570796325).
    
    The desired end‐effector (TCP) orientation is fixed (roll=0, pitch=0, yaw≈1.5708).
    To remove redundancy we further fix the wrist roll by setting q6 = 0.
    
    This function builds the forward kinematics as a sequence of rotation–translation steps,
    forms six nonlinear equations (three for position and three for orientation – only two of which
    are independent – but we choose a consistent set) and then uses Sympy’s nsolve to solve for
    the joint angles.
    
    :param p: A tuple (p_x, p_y, p_z) representing the target TCP position in the base frame.
    :return: A tuple (q1, q2, q3, q4, q5, q6) of joint angles (in radians) that reach the target.
    """
    import math
    import sympy as sp
    q1, q2, q3, q4, q5, q6 = sp.symbols('q1 q2 q3 q4 q5 q6', real=True)
    d2 = 0.13585
    off3 = -0.1197
    d3 = 0.425
    d4 = 0.39225
    off5 = 0.093
    d6 = 0.09465
    offTCP = 0.0823
    tcp_yaw = 1.570796325

    def RotZ(theta):
        return sp.Matrix([[sp.cos(theta), -sp.sin(theta), 0], [sp.sin(theta), sp.cos(theta), 0], [0, 0, 1]])

    def RotY(theta):
        return sp.Matrix([[sp.cos(theta), 0, sp.sin(theta)], [0, 1, 0], [-sp.sin(theta), 0, sp.cos(theta)]])
    p_total = sp.Matrix([0, 0, 0])
    R_total = sp.eye(3)
    R_total = RotZ(q1)
    p_total = p_total + R_total * sp.Matrix([0, d2, 0])
    R_total = R_total * RotY(q2)
    p_total = p_total + R_total * sp.Matrix([0, off3, d3])
    R_total = R_total * RotY(q3)
    p_total = p_total + R_total * sp.Matrix([0, 0, d4])
    R_total = R_total * RotY(q4)
    p_total = p_total + R_total * sp.Matrix([0, off5, 0])
    R_total = R_total * RotZ(q5)
    p_total = p_total + R_total * sp.Matrix([0, 0, d6])
    R_total = R_total * RotY(q6)
    p_total = p_total + R_total * sp.Matrix([0, offTCP, 0])
    R_total = R_total * RotZ(tcp_yaw)
    R_des = RotZ(tcp_yaw)
    px, py, pz = sp.symbols('px py pz', real=True)
    eq1 = sp.Eq(p_total[0], px)
    eq2 = sp.Eq(p_total[1], py)
    eq3 = sp.Eq(p_total[2], pz)
    eq4 = sp.Eq(R_total[0, 2], R_des[0, 2])
    eq5 = sp.Eq(R_total[1, 2], R_des[1, 2])
    eq6 = sp.Eq(q6, 0)
    eqs = [eq1, eq2, eq3, eq4, eq5, eq6]
    subs_dict = {px: p[0], py: p[1], pz: p[2]}
    sol = sp.nsolve([eq.subs(subs_dict) for eq in eqs], [q1, q2, q3, q4, q5, q6], (0, 0, 0, 0, 0, 0))
    q1_val, q2_val, q3_val, q4_val, q5_val, q6_val = [float(sol[i]) for i in range(6)]
    return (q1_val, q2_val, q3_val, q4_val, q5_val, q6_val)