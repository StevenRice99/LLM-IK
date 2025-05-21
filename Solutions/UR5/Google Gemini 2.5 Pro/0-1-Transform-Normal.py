import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    px, py, pz = p
    rx_T, ry_T, rz_T = r
    L1y = 0.13585
    L2y = -0.1197
    L2z = 0.425
    L_sum_y = L1y + L2y
    tol_angle = 1e-05

    def angle_normalize(angle: float) -> float:
        """Normalize angle to the range [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    def calculate_rpy_from_joints(j1: float, j2: float) -> tuple[float, float, float]:
        j2_normalized = angle_normalize(j2)
        calc_pitch = math.asin(math.sin(j2))
        calc_roll_raw: float
        calc_yaw_intermediate: float
        if abs(j2_normalized) <= math.pi / 2.0 + tol_angle:
            calc_roll_raw = 0.0
            calc_yaw_intermediate = j1
        else:
            calc_roll_raw = math.pi
            calc_yaw_intermediate = j1 + math.pi
        final_roll = angle_normalize(calc_roll_raw)
        final_yaw = angle_normalize(calc_yaw_intermediate)
        return (final_roll, calc_pitch, final_yaw)
    cos_theta2_val = pz / L2z
    if cos_theta2_val > 1.0:
        cos_theta2_val = 1.0
    elif cos_theta2_val < -1.0:
        cos_theta2_val = -1.0
    theta2_options = []
    th2_sol_acos = math.acos(cos_theta2_val)
    theta2_options.append(th2_sol_acos)
    if abs(th2_sol_acos) > tol_angle and abs(th2_sol_acos - math.pi) > tol_angle:
        theta2_options.append(-th2_sol_acos)
    elif abs(cos_theta2_val) == 1.0 and th2_sol_acos == 0.0:
        pass
    elif abs(cos_theta2_val) == 1.0 and abs(th2_sol_acos - math.pi) < tol_angle:
        theta2_options.append(-th2_sol_acos)
    rx_T_norm = angle_normalize(rx_T)
    ry_T_norm = angle_normalize(ry_T)
    rz_T_norm = angle_normalize(rz_T)
    for th2_cand in theta2_options:
        sin_th2_cand = math.sin(th2_cand)
        val_A_for_atan = L2z * sin_th2_cand
        val_B_for_atan = L_sum_y
        theta1_base = math.atan2(py, px) - math.atan2(val_B_for_atan, val_A_for_atan)
        th1_cand = angle_normalize(theta1_base)
        calc_r, calc_p, calc_y = calculate_rpy_from_joints(th1_cand, th2_cand)
        roll_match = False
        if abs(calc_r - rx_T_norm) < tol_angle:
            roll_match = True
        elif abs(abs(calc_r) - math.pi) < tol_angle and abs(abs(rx_T_norm) - math.pi) < tol_angle:
            roll_match = True
        pitch_match = abs(calc_p - ry_T_norm) < tol_angle
        yaw_match = abs(calc_y - rz_T_norm) < tol_angle
        if roll_match and pitch_match and yaw_match:
            return (th1_cand, th2_cand)
    return (0.0, 0.0)