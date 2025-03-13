import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    x_p, y_p, z_p = p
    A = 0.425
    B = 0.39225
    numerator = x_p ** 2 + z_p ** 2 - A ** 2 - B ** 2
    denominator = 2 * A * B
    if denominator == 0:
        return (0.0, 0.0, 0.0)
    cosθ2 = numerator / denominator
    cosθ2 = max(min(cosθ2, 1.0), -1.0)
    θ2_1 = math.acos(cosθ2)
    θ2_2 = -θ2_1
    solutions = []
    for θ2 in [θ2_1, θ2_2]:
        C = A + B * math.cos(θ2)
        D = B * math.sin(θ2)
        denominator_θ1 = x_p ** 2 + z_p ** 2
        if denominator_θ1 == 0:
            continue
        sinθ1 = (C * x_p - D * z_p) / denominator_θ1
        cosθ1 = (D * x_p + C * z_p) / denominator_θ1
        θ1 = math.atan2(sinθ1, cosθ1)
        x = A * math.sin(θ1) + B * math.sin(θ1 + θ2)
        z = A * math.cos(θ1) + B * math.cos(θ1 + θ2)
        error = (x - x_p) ** 2 + (z - z_p) ** 2
        solutions.append((error, θ1, θ2))
    if not solutions:
        return (0.0, 0.0, 0.0)
    solutions.sort()
    best_θ1, best_θ2 = (solutions[0][1], solutions[0][2])
    return (best_θ1, best_θ2, 0.0)