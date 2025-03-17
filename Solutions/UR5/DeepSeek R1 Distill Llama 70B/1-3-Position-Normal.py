import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    A = 0.39225
    B = 0.425
    p_x, _, p_z = p
    numerator = p_x ** 2 + p_z ** 2 - A ** 2 - B ** 2
    denominator = 2 * A * B
    cosθ2 = numerator / denominator
    cosθ2 = max(min(cosθ2, 1.0), -1.0)
    θ2 = math.acos(cosθ2)
    C = A * math.cos(θ2) + B
    D = A * math.sin(θ2)
    denominator_sin_cos = C ** 2 + D ** 2
    sinθ1 = (C * p_x - D * p_z) / denominator_sin_cos
    cosθ1 = (D * p_x + C * p_z) / denominator_sin_cos
    θ1 = math.atan2(sinθ1, cosθ1)
    θ3 = 0.0
    return (θ1, θ2, θ3)