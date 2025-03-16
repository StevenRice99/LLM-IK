import math
import numpy as np
from scipy.optimize import fsolve

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [roll, pitch, yaw].
    :return: A tuple of the values to set the joints to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    roll, pitch, yaw = r
    A = 0.09465
    B = 0.0823
    cosθ2 = (y - 0.093) / B
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

    def forward_kinematics(θ1, θ2, θ3):
        T1 = np.array([[math.cos(θ1), 0, math.sin(θ1), 0], [0, 1, 0, 0], [-math.sin(θ1), 0, math.cos(θ1), 0], [0, 0, 0, 1]])
        T2 = np.array([[math.cos(θ2), -math.sin(θ2), 0, 0], [math.sin(θ2), math.cos(θ2), 0, 0.093], [0, 0, 1, 0], [0, 0, 0, 1]])
        T3 = np.array([[math.cos(θ3), 0, math.sin(θ3), 0], [0, 1, 0, 0], [-math.sin(θ3), 0, math.cos(θ3), 0.09465], [0, 0, 0, 1]])
        T_tcp = np.array([[0, -1, 0, 0], [1, 0, 0, 0.0823], [0, 0, 1, 0], [0, 0, 0, 1]])
        T = T1 @ T2 @ T3 @ T_tcp
        roll = math.atan2(T[2, 1], T[2, 2])
        pitch = math.atan2(-T[2, 0], math.sqrt(T[2, 1] ** 2 + T[2, 2] ** 2))
        yaw = math.atan2(T[1, 0], T[0, 0])
        return (roll, pitch, yaw)

    def orientation_error(θ3):
        current_roll, current_pitch, current_yaw = forward_kinematics(θ1, θ2, θ3)
        error_roll = current_roll - roll
        error_pitch = current_pitch - pitch
        error_yaw = current_yaw - yaw
        return math.sqrt(error_roll ** 2 + error_pitch ** 2 + error_yaw ** 2)
    θ3_initial_guess = yaw
    θ3_solution = fsolve(orientation_error, θ3_initial_guess)
    θ3 = θ3_solution[0]
    return (θ1, θ2, θ3)