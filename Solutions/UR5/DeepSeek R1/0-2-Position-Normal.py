import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = p
    k = 0.01615
    k_sq = k ** 2
    numerator_cosθ3 = x ** 2 + y ** 2 + z ** 2 - 0.334746
    cosθ3 = numerator_cosθ3 / 0.3334125
    cosθ3 = max(min(cosθ3, 1.0), -1.0)
    θ3 = math.acos(cosθ3)
    C = 0.425 + 0.39225 * math.cos(θ3)
    D = 0.39225 * math.sin(θ3)
    A_sq = x ** 2 + y ** 2 - k_sq
    A = math.sqrt(A_sq) if A_sq >= 0 else 0.0
    denominator_theta2 = C ** 2 + D ** 2
    sinθ2 = (C * A - D * z) / denominator_theta2
    cosθ2 = (C * z + D * A) / denominator_theta2
    θ2 = math.atan2(sinθ2, cosθ2)
    denominator_theta1 = x ** 2 + y ** 2
    if denominator_theta1 == 0:
        θ1 = 0.0
    else:
        numerator_cosθ1 = A * x + k * y
        numerator_sinθ1 = A * y - k * x
        θ1 = math.atan2(numerator_sinθ1, numerator_cosθ1)
    return (θ1, θ2, θ3)