import math
import numpy as np

def rot_z_matrix(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)

def rot_y_matrix(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)

def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

class KinematicsSolver:
    d_J1_to_J2 = np.array([0.0, 0.13585, 0.0])
    d_J2_to_J3 = np.array([0.0, -0.1197, 0.425])
    d_J3_to_J4 = np.array([0.0, 0.0, 0.39225])
    d_J4_to_J5 = np.array([0.0, 0.093, 0.0])
    d_J5_to_J6 = np.array([0.0, 0.0, 0.09465])
    d_J6_to_TCP = np.array([0.0, 0.0823, 0.0])
    A_const = d_J6_to_TCP[1]
    B_const = d_J5_to_J6[2]
    V_J5_TCP_pre_th5 = np.array([0.0, A_const, B_const])

    def existing_inverse_kinematics(self, p_target_j6_origin: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
        x_w, y_w, z_w = p_target_j6_origin
        d2_y_link2_offset = self.d_J1_to_J2[1]
        y_chain_offset_in_j2_frame = self.d_J2_to_J3[1] + self.d_J4_to_J5[1]
        L1_planar = self.d_J2_to_J3[2]
        L2_planar = self.d_J3_to_J4[2]
        L3_planar = self.d_J5_to_J6[2]
        delta_for_t1_calc = y_chain_offset_in_j2_frame + d2_y_link2_offset
        r_xy_world = math.hypot(x_w, y_w)
        phi_xy_world = math.atan2(-x_w, y_w)
        arg_acos = 0.0
        if r_xy_world < 1e-09:
            arg_acos = 0.0 if abs(delta_for_t1_calc) < 1e-09 else 1.0 if delta_for_t1_calc > 0 else -1.0
        else:
            arg_acos = delta_for_t1_calc / r_xy_world
        arg_acos = max(-1.0, min(1.0, arg_acos))
        gamma_angle = math.acos(arg_acos)
        t1_candidates = [normalize_angle(phi_xy_world + gamma_angle), normalize_angle(phi_xy_world - gamma_angle)]
        best_solution_params = (float('inf'), 0.0, 0.0, 0.0, 0.0)
        for t1_val in t1_candidates:
            c1 = math.cos(t1_val)
            s1 = math.sin(t1_val)
            x_target_j2_frame = c1 * x_w + s1 * y_w
            y_coord_j2_frame = -s1 * x_w + c1 * y_w - d2_y_link2_offset
            z_target_j2_frame = z_w
            psi_angle_j2_xz = math.atan2(x_target_j2_frame, z_target_j2_frame)
            for T_orientation_L3_link in (psi_angle_j2_xz, normalize_angle(psi_angle_j2_xz + math.pi)):
                xw_planar_target = x_target_j2_frame - L3_planar * math.sin(T_orientation_L3_link)
                zw_planar_target = z_target_j2_frame - L3_planar * math.cos(T_orientation_L3_link)
                dist_sq_planar_target = xw_planar_target ** 2 + zw_planar_target ** 2
                epsilon_reach_sq = 1e-09
                if not (L1_planar - L2_planar) ** 2 - epsilon_reach_sq <= dist_sq_planar_target <= (L1_planar + L2_planar) ** 2 + epsilon_reach_sq:
                    if not (math.isclose(dist_sq_planar_target, (L1_planar + L2_planar) ** 2, rel_tol=1e-07, abs_tol=1e-07) or math.isclose(dist_sq_planar_target, (L1_planar - L2_planar) ** 2, rel_tol=1e-07, abs_tol=1e-07)):
                        continue
                den_cos_beta = 2 * L1_planar * L2_planar
                if abs(den_cos_beta) < 1e-12:
                    continue
                cos_beta_angle = (dist_sq_planar_target - L1_planar ** 2 - L2_planar ** 2) / den_cos_beta
                cos_beta_angle = max(-1.0, min(1.0, cos_beta_angle))
                for sign_beta in (+1.0, -1.0):
                    beta_val = sign_beta * math.acos(cos_beta_angle)
                    alpha_val = math.atan2(xw_planar_target, zw_planar_target)
                    den_delta_calc = L1_planar + L2_planar * math.cos(beta_val)
                    num_delta_calc = L2_planar * math.sin(beta_val)
                    delta_val = math.atan2(num_delta_calc, den_delta_calc)
                    t2_val = normalize_angle(alpha_val - delta_val)
                    t3_val = normalize_angle(beta_val)
                    t4_val = normalize_angle(T_orientation_L3_link - (t2_val + t3_val))
                    x_fk_check = L1_planar * math.sin(t2_val) + L2_planar * math.sin(t2_val + t3_val) + L3_planar * math.sin(t2_val + t3_val + t4_val)
                    z_fk_check = L1_planar * math.cos(t2_val) + L2_planar * math.cos(t2_val + t3_val) + L3_planar * math.cos(t2_val + t3_val + t4_val)
                    current_err_sq_val = (x_fk_check - x_target_j2_frame) ** 2 + (z_fk_check - z_target_j2_frame) ** 2 + (y_coord_j2_frame - y_chain_offset_in_j2_frame) ** 2
                    if current_err_sq_val < best_solution_params[0]:
                        best_solution_params = (current_err_sq_val, t1_val, t2_val, t3_val, t4_val)
        _, final_t1, final_t2, final_t3, final_t4 = best_solution_params
        return (final_t1, final_t2, final_t3, final_t4, 0.0)

    def _fk_to_J5_origin_and_R04(self, th1, th2, th3, th4):
        p_J1_w = np.array([0.0, 0.0, 0.0])
        R_0_1 = rot_z_matrix(th1)
        p_J2_w = p_J1_w + R_0_1 @ self.d_J1_to_J2
        R_0_2 = R_0_1 @ rot_y_matrix(th2)
        p_J3_w = p_J2_w + R_0_2 @ self.d_J2_to_J3
        R_0_3 = R_0_2 @ rot_y_matrix(th3)
        p_J4_w = p_J3_w + R_0_3 @ self.d_J3_to_J4
        R_0_4 = R_0_3 @ rot_y_matrix(th4)
        p_J5_origin_w = p_J4_w + R_0_4 @ self.d_J4_to_J5
        return (p_J5_origin_w, R_0_4)

    def inverse_kinematics(self, p_tcp_target_tuple: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
        p_tcp_desired_np = np.array(p_tcp_target_tuple, dtype=float)
        th5 = 0.0
        th1, th2, th3, th4, _ = self.existing_inverse_kinematics(p_tcp_target_tuple)
        N_ITERATIONS = 15
        for _ in range(N_ITERATIONS):
            p_J5_origin_curr, R_0_4_curr = self._fk_to_J5_origin_and_R04(th1, th2, th3, th4)
            s5 = math.sin(th5)
            c5 = math.cos(th5)
            X_L4_calc = np.array([-self.A_const * s5, self.A_const * c5, 0.0])
            p_V_target_np = p_tcp_desired_np - R_0_4_curr @ X_L4_calc
            th1, th2, th3, th4, _ = self.existing_inverse_kinematics(tuple(p_V_target_np.tolist()))
            p_J5_origin_new, R_0_4_new = self._fk_to_J5_origin_and_R04(th1, th2, th3, th4)
            LHS_J4_frame = R_0_4_new.T @ (p_tcp_desired_np - p_J5_origin_new)
            if self.A_const == 0:
                th5 = 0.0
            else:
                s5_component_num = -LHS_J4_frame[0]
                c5_component_num = LHS_J4_frame[1]
                th5 = math.atan2(s5_component_num, c5_component_num)
        th6_final = 0.0
        return (normalize_angle(th1), normalize_angle(th2), normalize_angle(th3), normalize_angle(th4), normalize_angle(th5), normalize_angle(th6_final))

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    solver = KinematicsSolver()
    return solver.inverse_kinematics(p)