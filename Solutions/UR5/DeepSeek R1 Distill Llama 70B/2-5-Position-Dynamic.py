import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    A = 0.09465
    B = 0.0823
    cos_theta2 = (y - 0.093) / B
    theta2 = math.acos(cos_theta2)
    sin_theta2 = math.sin(theta2)
    M = [[A, -B * sin_theta2], [B * sin_theta2, A]]
    det_M = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    if det_M == 0:
        raise ValueError('Matrix is singular; target position is not reachable.')
    inv_det_M = 1.0 / det_M
    inv_M = [[inv_det_M * M[1][1], -inv_det_M * M[0][1]], [-inv_det_M * M[1][0], inv_det_M * M[0][0]]]
    sin_theta1 = inv_M[0][0] * x + inv_M[0][1] * z
    cos_theta1 = inv_M[1][0] * x + inv_M[1][1] * z
    sin_theta1 = max(min(sin_theta1, 1.0), -1.0)
    cos_theta1 = max(min(cos_theta1, 1.0), -1.0)
    theta1 = math.atan2(sin_theta1, cos_theta1)
    theta3 = 0.0
    x3 = x
    z3 = z
    desired_x = p[0]
    desired_z = p[2]
    A = x3
    B = z3
    C = desired_x
    D = desired_z
    denominator = A ** 2 + B ** 2
    if denominator == 0:
        raise ValueError('Denominator is zero; cannot solve for theta4.')
    cos_theta4 = (A * C + B * D) / denominator
    sin_theta4 = (A * D - B * C) / denominator
    theta4 = math.atan2(sin_theta4, cos_theta4)
    return (theta1, theta2, theta3, theta4)