import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Analytic 2‑DOF IK for:
       joint1: revolute about Z at origin
       joint2: revolute about Y, offset +0.13585 in Y
       TCP:    offset [0, –0.1197, 0.425] in joint2 frame

    We solve p and the desired orientation r = [roll, pitch, yaw].
    Since we only have 2 DOF, we can only match pitch and yaw.  Roll
    on the TCP will always be 0 or π, so we ignore it in our error metric.

    :param p: goal TCP position (x,y,z)
    :param r: goal TCP orientation (roll, pitch, yaw)
    :return: (θ1, θ2)
    """
    x, y, z = p
    _, pitch_desired, yaw_desired = r
    a2 = 0.13585
    a3y = -0.1197
    d3 = 0.425
    cos_th2 = z / d3
    cos_th2 = max(min(cos_th2, 1.0), -1.0)
    th2_pos = math.acos(cos_th2)
    th2_neg = -th2_pos
    candidates = []
    for th2 in (th2_pos, th2_neg):
        vx = d3 * math.sin(th2)
        vy = a2 + a3y
        phi_v = math.atan2(vy, vx)
        phi_p = math.atan2(y, x)
        th1 = phi_p - phi_v
        th1 = math.atan2(math.sin(th1), math.cos(th1))
        th2 = math.atan2(math.sin(th2), math.cos(th2))
        R11 = math.cos(th1) * math.cos(th2)
        R21 = math.sin(th1) * math.cos(th2)
        R31 = -math.sin(th2)
        pitch_map = math.atan2(-R31, math.hypot(R11, R21))
        yaw_map = math.atan2(R21, R11)
        err = (pitch_map - pitch_desired) ** 2 + (yaw_map - yaw_desired) ** 2
        candidates.append((err, (th1, th2)))
    _, (theta1, theta2) = min(candidates, key=lambda c: c[0])
    return (theta1, theta2)