import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values (q1,q2,q3) to reach position p=[x,y,z]
    and orientation r=[roll,pitch,yaw] for the 3‑DOF Z–Y–Y arm:
       joint1: revolute about Z
       joint2: revolute about Y, offset   [0, +0.13585, 0]
       joint3: revolute about Y, offset   [0, -0.11970, +0.425]
       TCP:                            [0,  0,     +0.39225]
    """
    x, y, z = p
    roll, pitch, yaw = r
    eps = 1e-08
    if abs(roll) < eps:
        pitch2 = pitch
        yaw2 = yaw
    elif abs(abs(roll) - math.pi) < eps:
        pitch2 = math.pi - pitch
        yaw2 = yaw + math.copysign(math.pi, roll)
    else:
        raise ValueError(f'Roll={roll:.6g} unsupported for a Z–Y–Y chain')

    def wrap(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi
    pitch2 = wrap(pitch2)
    yaw2 = wrap(yaw2)
    q1 = yaw2
    A = math.cos(q1) * x + math.sin(q1) * y
    Z = z
    L2 = 0.425
    L3 = 0.39225
    R2 = A * A + Z * Z
    c3 = (R2 - L2 * L2 - L3 * L3) / (2.0 * L2 * L3)
    c3 = max(-1.0, min(1.0, c3))
    γ = math.acos(c3)
    best = (1000000000.0, 0.0, 0.0)
    for q3 in (γ, -γ):
        k1 = L2 + L3 * math.cos(q3)
        k2 = L3 * math.sin(q3)
        q2 = math.atan2(A * k1 - Z * k2, Z * k1 + A * k2)
        err = abs(wrap(q2 + q3 - pitch2))
        if err < best[0]:
            best = (err, q2, q3)
    _, q2, q3 = best
    return (q1, q2, q3)