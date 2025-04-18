import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Inverse kinematics for the 2‑DOF serial arm:
      Joint1: revolute about Z
         link1 offset: [0, 0.13585, 0]
      Joint2: revolute about Y
         TCP offset:   [0, -0.1197, 0.425]
    :param p: desired TCP position (x, y, z)
    :param r: desired TCP orientation in radians as (roll, pitch, yaw)
             using the same roll–pitch–yaw convention as the URDF
    :return: (theta1, theta2)
    """
    x, y, z = p
    roll, pitch, yaw = r
    tol = 1e-07
    pi = math.pi

    def wrap(a: float) -> float:
        return (a + pi) % (2 * pi) - pi
    B = 0.13585 - 0.1197
    rw = wrap(roll)
    if abs(rw) < tol:
        theta1 = wrap(yaw)
        theta2 = wrap(pitch)
    elif abs(abs(rw) - pi) < tol:
        s = 1.0 if rw > 0 else -1.0
        theta1 = wrap(yaw - s * pi)
        theta2 = wrap(pi - pitch)
    else:
        cos_t2 = z / 0.425
        cos_t2 = max(-1.0, min(1.0, cos_t2))
        t2a = math.acos(cos_t2)
        t2b = -t2a
        best_err = float('inf')
        best = (0.0, 0.0)
        for t2 in (t2a, t2b):
            A = 0.425 * math.sin(t2)
            D = A * A + B * B
            C = (A * x + B * y) / D
            S = (-B * x + A * y) / D
            t1 = math.atan2(S, C)
            x_fk = A * math.cos(t1) - B * math.sin(t1)
            y_fk = A * math.sin(t1) + B * math.cos(t1)
            err = (x_fk - x) ** 2 + (y_fk - y) ** 2
            if err < best_err:
                best_err = err
                best = (t1, t2)
        theta1, theta2 = best
    return (theta1, theta2)