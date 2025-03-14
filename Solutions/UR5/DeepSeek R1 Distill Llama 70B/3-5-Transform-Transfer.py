import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [rx, ry, rz].
    :return: A tuple of the values to set the joints to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    A = 0.09465
    B = 0.0823
    cosθ2 = (y - 0.093) / 0.0823
    θ2 = math.acos(cosθ2)
    sinθ2 = math.sin(θ2)
    M = np.array([[A, -B * sinθ2], [B * sinθ2, A]], dtype=float)
    det_M = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if det_M == 0:
        raise ValueError('Matrix is singular; target position is not reachable.')
    inv_det_M = 1.0 / det_M
    inv_M = np.array([[inv_det_M * M[1, 1], -inv_det_M * M[0, 1]], [-inv_det_M * M[1, 0], inv_det_M * M[0, 0]]], dtype=float)
    sinθ1 = inv_M[0, 0] * x + inv_M[0, 1] * z
    cosθ1 = inv_M[1, 0] * x + inv_M[1, 1] * z
    sinθ1 = max(min(sinθ1, 1.0), -1.0)
    cosθ1 = max(min(cosθ1, 1.0), -1.0)
    θ1 = math.atan2(sinθ1, cosθ1)
    R1 = np.array([[math.cos(θ1), 0, math.sin(θ1)], [0, 1, 0], [-math.sin(θ1), 0, math.cos(θ1)]], dtype=float)
    R2 = np.array([[math.cos(θ2), -math.sin(θ2), 0], [math.sin(θ2), math.cos(θ2), 0], [0, 0, 1]], dtype=float)
    R_current = np.dot(R2, R1)
    rx, ry, rz = r
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]], dtype=float)
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]], dtype=float)
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]], dtype=float)
    R_desired = np.dot(Rz, np.dot(Ry, Rx))
    R_remaining = np.dot(R_desired, R_current.T)
    trace = np.trace(R_remaining)
    if trace > 4.0:
        trace = 4.0
    elif trace < -2.0:
        trace = -2.0
    theta = math.acos((trace - 1.0) / 2.0)
    if abs(theta) < 1e-06:
        θ3 = 0.0
    else:
        axis = np.array([R_remaining[1, 2] - R_remaining[2, 1], R_remaining[2, 0] - R_remaining[0, 2], R_remaining[0, 1] - R_remaining[1, 0]])
        axis = axis / np.linalg.norm(axis)
        θ3 = theta * axis[1]
    return (θ1, θ2, θ3)