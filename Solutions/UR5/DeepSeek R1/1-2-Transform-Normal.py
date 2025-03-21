import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    x_p, y_p, z_p = p
    roll, pitch, yaw = r

    def is_pi(angle):
        return abs((angle + math.pi) % (2 * math.pi) - math.pi) < 1e-06
    if is_pi(roll) and is_pi(yaw):
        x_p = -x_p
        z_p = -z_p
    a = 0.39225
    b = 0.425
    d_sq = x_p ** 2 + z_p ** 2
    numerator = d_sq - a ** 2 - b ** 2
    denominator = 2 * a * b
    cos_q2 = numerator / denominator
    cos_q2 = max(min(cos_q2, 1.0), -1.0)
    q2_1 = math.acos(cos_q2)
    q2_2 = -q2_1

    def compute_q1(q2):
        A = x_p - a * math.sin(pitch)
        B = z_p - a * math.cos(pitch)
        q1 = math.atan2(A - b * math.sin(q2), B - b * math.cos(q2))
        return q1
    q1_1 = compute_q1(q2_1)
    q1_2 = compute_q1(q2_2)
    error1 = abs((q1_1 + q2_1 - pitch + math.pi) % (2 * math.pi) - math.pi)
    error2 = abs((q1_2 + q2_2 - pitch + math.pi) % (2 * math.pi) - math.pi)
    if error1 < error2:
        q1, q2 = (q1_1, q2_1)
    else:
        q1, q2 = (q1_2, q2_2)
    q1 = (q1 + math.pi) % (2 * math.pi) - math.pi
    q2 = (q2 + math.pi) % (2 * math.pi) - math.pi
    return (q1, q2)