import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [roll, pitch, yaw].
    :return: A tuple of the values to set the joints to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    roll, pitch, yaw = r
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
    R_roll = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    R_pitch = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    R_yaw = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    R_desired = R_yaw @ R_pitch @ R_roll
    R_joint1 = np.array([[math.cos(θ1), 0, math.sin(θ1)], [0, 1, 0], [-math.sin(θ1), 0, math.cos(θ1)]])
    R_joint2 = np.array([[math.cos(θ2), -math.sin(θ2), 0], [math.sin(θ2), math.cos(θ2), 0], [0, 0, 1]])
    R_TCP = R_joint1 @ R_joint2
    θ1 += yaw
    θ2 += pitch
    return (θ1, θ2)