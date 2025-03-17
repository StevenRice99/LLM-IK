import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    p_x, p_y, p_z = p
    if abs(p_z) > 0.425:
        raise ValueError('Target z-coordinate out of reach')
    θ2 = math.acos(p_z / 0.425)
    if p_z < 0:
        θ2 = -θ2
    sinθ2 = math.sin(θ2)
    A = 0.425 * sinθ2
    B = -0.01615
    Δ = A ** 2 + B ** 2
    cosθ1 = (A * p_x - B * p_y) / Δ
    sinθ1 = (B * p_x + A * p_y) / Δ
    θ1 = math.atan2(sinθ1, cosθ1)
    return (θ1, θ2)