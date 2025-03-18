def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    import math
    L1 = 0.13585
    L2 = 0.1197
    L3 = 0.425
    px, py, pz = p
    rx, ry, rz = r
    q2 = math.acos(pz / L3)
    s2 = math.sin(q2)
    if s2 == 0:
        if pz == L3:
            q2 = 0.0
        else:
            q2 = math.pi
        if py == 0 and px == 0:
            q1 = 0.0
        else:
            q1 = rz - q2
    else:
        q1 = rz - q2
    return (q1, q2)