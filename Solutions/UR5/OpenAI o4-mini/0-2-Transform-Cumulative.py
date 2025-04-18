import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Analytic IK for the 3‑DOF arm:
      Joint1: revolute about Z at the base
      Joint2: revolute about Y with +0.13585\u2009m Y‑offset
      Joint3: revolute about Y with +0.425\u2009m Z‑offset, then TCP at +0.39225\u2009m Z

    :param p: target TCP position [x, y, z] in base frame
    :param r: target TCP orientation [roll, pitch, yaw] in radians (extrinsic Rz*Ry*Rx)
    :returns: (q1, q2, q3)
    """
    x, y, z = p
    rx, ry, rz = r

    def wrap(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))
    d4 = 0.39225
    tol = 1e-06
    if abs(abs(rx) - math.pi) < tol:
        q1 = wrap(rz + math.pi)
        q23 = wrap(math.pi - ry)
    else:
        q1 = wrap(rz)
        q23 = wrap(ry)
    c1 = math.cos(q1)
    s1 = math.sin(q1)
    xp = c1 * x + s1 * y
    zp = z
    xp2 = xp - d4 * math.sin(q23)
    zp2 = zp - d4 * math.cos(q23)
    q2 = wrap(math.atan2(xp2, zp2))
    q3 = wrap(q23 - q2)
    return (q1, q2, q3)