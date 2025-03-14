import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    x, y, z = p
    cosθ2 = z / 0.425
    cosθ2 = max(min(cosθ2, 1.0), -1.0)
    θ2 = math.acos(cosθ2)
    sinθ2 = math.sqrt(1 - cosθ2 ** 2)
    if z < 0:
        θ2 = -θ2
    cosθ1 = (y + 0.1197 * (z / 0.425)) / 0.13585
    cosθ1 = max(min(cosθ1, 1.0), -1.0)
    θ1 = math.acos(cosθ1)
    sinθ1 = math.sqrt(1 - cosθ1 ** 2)
    if y < -0.1197 * (z / 0.425):
        θ1 = -θ1
    x_calculated = -0.13585 * math.sin(θ1) - 0.425 * math.sin(θ2)
    if abs(x_calculated - x) > 0.001:
        for _ in range(10):
            error_x = x - x_calculated
            θ1 += 0.1 * error_x
            θ2 += 0.1 * error_x
            cosθ1 = (y + 0.1197 * (z / 0.425)) / 0.13585
            cosθ1 = max(min(cosθ1, 1.0), -1.0)
            θ1 = math.acos(cosθ1)
            cosθ2 = z / 0.425
            cosθ2 = max(min(cosθ2, 1.0), -1.0)
            θ2 = math.acos(cosθ2)
            sinθ1 = math.sqrt(1 - cosθ1 ** 2)
            if y < -0.1197 * (z / 0.425):
                θ1 = -θ1
            sinθ2 = math.sqrt(1 - cosθ2 ** 2)
            if z < 0:
                θ2 = -θ2
            x_calculated = -0.13585 * math.sin(θ1) - 0.425 * math.sin(θ2)
    return (θ1, θ2)