import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    cos_theta3 = (y - 0.093) / 0.0823
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = math.acos(cos_theta3)
    sin_theta3 = math.sin(theta3)
    theta4 = 0.0
    A = 0.093
    B = 0.09465
    cos_theta2 = y / A
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    sin_theta2 = math.sqrt(1 - cos_theta2 ** 2)
    M = [[-A * sin_theta2, B], [B, A * sin_theta2]]
    det_M = -A * sin_theta2 * (A * sin_theta2) - B * B
    if det_M == 0:
        raise ValueError('Matrix is singular; target position is not reachable.')
    inv_det_M = 1.0 / det_M
    inv_M = [[inv_det_M * (A * sin_theta2), -inv_det_M * B], [-inv_det_M * B, inv_det_M * (-A * sin_theta2)]]
    cos_theta1 = inv_M[0][0] * x + inv_M[0][1] * z
    sin_theta1 = inv_M[1][0] * x + inv_M[1][1] * z
    cos_theta1 = max(min(cos_theta1, 1.0), -1.0)
    sin_theta1 = max(min(sin_theta1, 1.0), -1.0)
    theta1 = math.atan2(sin_theta1, cos_theta1)
    theta2 = math.acos(cos_theta2)
    return (theta1, theta2, theta3, theta4)