import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [roll, pitch, yaw] (Rx, Ry, Rz).
    :return: A tuple of the joint values (j1, j2, j3, j4).
    """
    px, py, pz_world = p
    roll, pitch, yaw = r
    Y_req = 0.10915
    L1 = 0.425
    L2 = 0.39225
    H = math.sqrt(px ** 2 + py ** 2)
    val_for_acos_phi = 0.0
    if H < 1e-09:
        if abs(Y_req) < 1e-09:
            val_for_acos_phi = 0.0
        else:
            val_for_acos_phi = np.clip(Y_req / (H + 1e-12), -1.0, 1.0)
    else:
        val_for_acos_phi = np.clip(Y_req / H, -1.0, 1.0)
    phi_offset = math.acos(val_for_acos_phi)
    j1_base_angle = math.atan2(-px, py)
    j1 = j1_base_angle - phi_offset
    cj1 = math.cos(j1)
    sj1 = math.sin(j1)
    x_planar = px * cj1 + py * sj1
    z_planar = pz_world
    d_sq = x_planar ** 2 + z_planar ** 2
    cos_j3_val_num = d_sq - L1 ** 2 - L2 ** 2
    cos_j3_val_den = 2 * L1 * L2
    cos_j3_val = 0.0
    if abs(cos_j3_val_den) < 1e-12:
        cos_j3_val = np.clip(cos_j3_val_num / (cos_j3_val_den + 1e-12 if cos_j3_val_den >= 0 else -1e-12), -1.0, 1.0)
    else:
        cos_j3_val = np.clip(cos_j3_val_num / cos_j3_val_den, -1.0, 1.0)
    j3_raw = -math.acos(cos_j3_val)
    s_j3_raw = math.sin(j3_raw)
    c_j3_raw = math.cos(j3_raw)
    k1 = L1 + L2 * c_j3_raw
    k2 = L2 * s_j3_raw
    sin_j2 = 0.0
    cos_j2 = 1.0
    if d_sq > 1e-09:
        sin_j2 = (k1 * x_planar - k2 * z_planar) / d_sq
        cos_j2 = (k2 * x_planar + k1 * z_planar) / d_sq
    j2 = math.atan2(sin_j2, cos_j2)
    cr_world, sr_world = (math.cos(roll), math.sin(roll))
    cp_world, sp_world = (math.cos(pitch), math.sin(pitch))
    cy_world, sy_world = (math.cos(yaw), math.sin(yaw))
    R_target_world = np.array([[cy_world * cp_world, cy_world * sp_world * sr_world - sy_world * cr_world, cy_world * sp_world * cr_world + sy_world * sr_world], [sy_world * cp_world, sy_world * sp_world * sr_world + cy_world * cr_world, sy_world * sp_world * cr_world - cy_world * sr_world], [-sp_world, cp_world * sr_world, cp_world * cr_world]])
    R_01_T = np.array([[cj1, sj1, 0], [-sj1, cj1, 0], [0, 0, 1]])
    R_j2_base = R_01_T @ R_target_world
    sum_angles_j234 = math.atan2(R_j2_base[0, 2], R_j2_base[0, 0])
    j4 = sum_angles_j234 - j2 - j3_raw
    j1_norm = (j1 + math.pi) % (2 * math.pi) - math.pi
    j2_norm = (j2 + math.pi) % (2 * math.pi) - math.pi
    j3_norm = (j3_raw + math.pi) % (2 * math.pi) - math.pi
    j4_norm = (j4 + math.pi) % (2 * math.pi) - math.pi
    return (j1_norm, j2_norm, j3_norm, j4_norm)