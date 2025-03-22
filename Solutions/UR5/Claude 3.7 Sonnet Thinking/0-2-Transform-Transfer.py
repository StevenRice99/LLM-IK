import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    rx, ry, rz = r
    θ1 = rz
    c1, s1 = (math.cos(θ1), math.sin(θ1))
    x_1 = c1 * x + s1 * y
    y_1 = -s1 * x + c1 * y
    z_1 = z
    d2 = 0.13585
    d3 = -0.1197
    a2 = 0.425
    a3 = 0.39225
    y_1 = y_1 - d2
    r_sq = x_1 ** 2 + y_1 ** 2 + z_1 ** 2
    cos_θ3 = (r_sq - a2 ** 2 - a3 ** 2 - d3 ** 2) / (2 * a2 * math.sqrt(a3 ** 2 + d3 ** 2))
    cos_θ3 = max(min(cos_θ3, 1.0), -1.0)
    θ3_1 = math.acos(cos_θ3) - math.atan2(d3, a3)
    θ3_2 = -math.acos(cos_θ3) - math.atan2(d3, a3)

    def calculate_θ2(θ3):
        x3 = a2 + math.cos(θ3) * a3 - math.sin(θ3) * d3
        y3 = math.sin(θ3) * a3 + math.cos(θ3) * d3
        denominator = math.sqrt(x3 ** 2 + y3 ** 2)
        if denominator < 1e-06:
            return 0.0
        sin_θ2 = (x3 * z_1 - y3 * math.sqrt(x_1 ** 2 + y_1 ** 2)) / denominator
        cos_θ2 = (x3 * math.sqrt(x_1 ** 2 + y_1 ** 2) + y3 * z_1) / denominator
        return math.atan2(sin_θ2, cos_θ2)
    θ2_1 = calculate_θ2(θ3_1)
    θ2_2 = calculate_θ2(θ3_2)
    solutions = [(θ1, θ2_1, θ3_1), (θ1, θ2_2, θ3_2)]
    best_solution = None
    min_error = float('inf')
    for sol in solutions:
        _, θ2, θ3 = sol
        error = abs(θ2 + θ3 - ry)
        if error < min_error:
            min_error = error
            best_solution = sol
    return best_solution