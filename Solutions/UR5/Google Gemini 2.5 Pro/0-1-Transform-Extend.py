import math
L1_Y_OFFSET = 0.13585
TCP_Y_OFFSET_IN_L2 = -0.1197
TCP_Z_OFFSET_IN_L2 = 0.425
S_PARAM = TCP_Z_OFFSET_IN_L2
A_PARAM = L1_Y_OFFSET + TCP_Y_OFFSET_IN_L2

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [px, py, pz].
    :param r: The orientation to reach in radians in the form [roll, pitch, yaw].
    :return: A tuple (joint1_angle, joint2_angle) in radians.
    """
    px, py, pz = p
    r_roll, r_pitch, r_yaw = r
    cos_theta2_val = pz / S_PARAM
    if cos_theta2_val > 1.0:
        cos_theta2_val = 1.0
    elif cos_theta2_val < -1.0:
        cos_theta2_val = -1.0
    theta2_abs = math.acos(cos_theta2_val)
    sin_r_pitch = math.sin(r_pitch)
    chosen_theta2: float
    if sin_r_pitch < -1e-12:
        chosen_theta2 = -theta2_abs
    else:
        chosen_theta2 = theta2_abs
    sin_chosen_theta2 = math.sin(chosen_theta2)
    k1 = S_PARAM * sin_chosen_theta2
    k2 = A_PARAM
    sin_theta1_component = k1 * py - k2 * px
    cos_theta1_component = k1 * px + k2 * py
    final_theta1 = math.atan2(sin_theta1_component, cos_theta1_component)
    return (final_theta1, chosen_theta2)