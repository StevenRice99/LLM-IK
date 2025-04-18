import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    A simple closed‐form “one‐branch” IK that chooses θ3=0, θ4=0, θ6=0, and solves θ5, θ2, θ1
    so that the full 6‑DOF chain Z–Y–Y–Y–Z–Y reaches the given position p.
    This is an analytic O(1) computation (no iterative loops over unknowns).
    """
    x, y, z = p
    d2 = 0.13585
    d23_y = -0.1197
    d23_z = 0.425
    d34_z = 0.39225
    d45_y = 0.093
    d56_z = 0.09465
    d6E_y = 0.0823
    a = d2 + d23_y + d45_y
    D = d34_z + d56_z
    C0 = d23_z + D
    K = d6E_y
    norm2 = x * x + y * y + z * z
    num = norm2 - (K * K + C0 * C0 + a * a)
    den = 2.0 * a * K
    c5 = num / den
    if c5 > 1.0:
        c5 = 1.0
    if c5 < -1.0:
        c5 = -1.0
    theta5 = math.acos(c5)
    A = C0
    B = K * math.sin(theta5)
    L = math.hypot(A, B)
    phi = math.atan2(B, A)
    arg = z / L
    if arg > 1.0:
        arg = 1.0
    if arg < -1.0:
        arg = -1.0
    theta2 = phi - math.acos(arg)
    X2 = -K * math.sin(theta5)
    Z2 = A
    Vx = math.cos(theta2) * X2 + math.sin(theta2) * Z2
    Vy = a + K * math.cos(theta5)
    num1 = y * Vx - x * Vy
    den1 = x * Vx + y * Vy
    theta1 = math.atan2(num1, den1)
    theta3 = 0.0
    theta4 = 0.0
    theta6 = 0.0

    def norm_ang(u: float) -> float:
        while u > math.pi:
            u -= 2 * math.pi
        while u < -math.pi:
            u += 2 * math.pi
        return u
    return (norm_ang(theta1), norm_ang(theta2), norm_ang(theta3), norm_ang(theta4), norm_ang(theta5), norm_ang(theta6))