import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = p
    link2_offset = 0.13585
    link3_offset = 0.1197
    tcp_offset = 0.39225
    z_target = z - 0.425
    cos_q2 = z_target / tcp_offset
    cos_q2 = max(min(cos_q2, 1.0), -1.0)
    q2 = math.acos(cos_q2)
    sin_q2 = math.sqrt(1 - cos_q2 ** 2)
    B = tcp_offset * sin_q2
    A = link2_offset + link3_offset
    det = B ** 2 - A ** 2
    if det == 0:
        q1 = 0.0
    else:
        cos_q1 = (B * x - A * y) / det
        sin_q1 = (-A * x - B * y) / det
        cos_q1 = max(min(cos_q1, 1.0), -1.0)
        sin_q1 = max(min(sin_q1, 1.0), -1.0)
        q1 = math.atan2(sin_q1, cos_q1)
    q3 = 0.0
    return (q1, q2, q3)