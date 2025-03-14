import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    A = 0.093
    B = 0.09465
    cosθ2 = y / A
    sinθ2 = math.sqrt(1 - cosθ2 ** 2)
    M = [[-A * sinθ2, B], [B, A * sinθ2]]
    det_M = -A * sinθ2 * (A * sinθ2) - B * B
    if det_M == 0:
        raise ValueError('Matrix is singular; target position is not reachable.')
    inv_det_M = 1.0 / det_M
    inv_M = [[inv_det_M * (A * sinθ2), -inv_det_M * B], [-inv_det_M * B, inv_det_M * (-A * sinθ2)]]
    cosθ1 = inv_M[0][0] * x + inv_M[0][1] * z
    sinθ1 = inv_M[1][0] * x + inv_M[1][1] * z
    cosθ1 = max(min(cosθ1, 1.0), -1.0)
    sinθ1 = max(min(sinθ1, 1.0), -1.0)
    θ1 = math.atan2(sinθ1, cosθ1)
    θ2 = math.acos(cosθ2)
    return (θ1, θ2)