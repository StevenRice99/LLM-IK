def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Closed-form inverse kinematics solution for a 4-DOF arm:
      • Joint 1 (q1) rotates about Z 
      • Joints 2, 3, 4 (q2, q3, q4) rotate about Y
    Link positions from DETAILS (offsets are relatively small in Y but are ignored here for simplicity).
    Orientation "r" = (roll, pitch, yaw) is interpreted so that:
      q1 ≈ yaw (rotation about Z),
      q2 + q3 + q4 ≈ pitch (net rotation about Y),
      and roll is assumed to be 0 for the feasible solutions (the robot cannot independently roll its TCP).

    This closed-form avoids any iterative or symbolic solvers to prevent timeouts.
    It should perform adequately for most reachable targets under the assumption that
    ignoring small link offsets in Y for the geometry is acceptable here.
    """
    import math
    x, y, z = p
    roll, pitch, yaw = r
    q1 = yaw
    Xp = x * math.cos(-q1) - y * math.sin(-q1)
    L2 = 0.425
    L3 = 0.39225
    d = math.hypot(Xp, z)
    cos_q3 = (d * d - L2 * L2 - L3 * L3) / (2.0 * L2 * L3)
    cos_q3 = max(min(cos_q3, 1.0), -1.0)
    q3_elbow = math.acos(cos_q3)
    q3 = -q3_elbow
    alpha = math.atan2(z, Xp)
    sin_q3 = math.sin(q3)
    cos_q3 = math.cos(q3)
    beta = math.atan2(L3 * sin_q3, L2 + L3 * cos_q3)
    q2 = alpha - beta
    q4 = pitch - (q2 + q3)

    def wrap_angle(angle):
        wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
        return wrapped
    q1 = wrap_angle(q1)
    q2 = wrap_angle(q2)
    q3 = wrap_angle(q3)
    q4 = wrap_angle(q4)
    return (q1, q2, q3, q4)