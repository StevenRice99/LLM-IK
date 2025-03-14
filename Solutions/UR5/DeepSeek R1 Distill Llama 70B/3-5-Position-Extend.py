import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    A = 0.09465
    B = 0.0823
    cosθ2 = (y - 0.093) / 0.0823
    θ2 = math.acos(cosθ2)
    sinθ2 = math.sin(θ2)
    M = [[A, -B * sinθ2], [B * sinθ2, A]]
    det_M = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    if det_M == 0:
        raise ValueError('Matrix is singular; target position is not reachable.')
    inv_det_M = 1.0 / det_M
    inv_M = [[inv_det_M * M[1][1], -inv_det_M * M[0][1]], [-inv_det_M * M[1][0], inv_det_M * M[0][0]]]
    sinθ1 = inv_M[0][0] * x + inv_M[0][1] * z
    cosθ1 = inv_M[1][0] * x + inv_M[1][1] * z
    sinθ1 = max(min(sinθ1, 1.0), -1.0)
    cosθ1 = max(min(cosθ1, 1.0), -1.0)
    θ1 = math.atan2(sinθ1, cosθ1)
    θ3 = 0.0
    return (θ1, θ2, θ3)