import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [roll, pitch, yaw].
    :return: A tuple of the joint values (q1, q2, q3) in radians.
    """
    px, py, pz = p
    roll, pitch, yaw = (r[0], r[1], r[2])
    l2 = 0.425
    l3 = 0.39225
    J_MIN = -2.0 * math.pi
    J_MAX = 2.0 * math.pi
    epsilon_gimbal = 1e-06
    epsilon_limit = 1e-09
    q1_raw = 0.0
    q_sum_raw = 0.0
    if abs(math.cos(pitch)) < epsilon_gimbal:
        q_sum_raw = pitch
        if pitch > 0:
            q1_raw = yaw - roll
        else:
            q1_raw = yaw + roll
    else:
        current_roll_normalized = math.atan2(math.sin(roll), math.cos(roll))
        if abs(current_roll_normalized) < math.pi / 4.0:
            q1_raw = yaw
            q_sum_raw = pitch
        else:
            q1_raw = yaw + math.pi
            q_sum_raw = math.pi - pitch
    q1 = math.atan2(math.sin(q1_raw), math.cos(q1_raw))
    q_sum = math.atan2(math.sin(q_sum_raw), math.cos(q_sum_raw))
    cos_q1 = math.cos(q1)
    sin_q1 = math.sin(q1)
    cos_q_sum = math.cos(q_sum)
    sin_q_sum = math.sin(q_sum)
    term_Y_for_atan2 = cos_q1 * px + sin_q1 * py - l3 * sin_q_sum
    term_X_for_atan2 = pz - l3 * cos_q_sum
    q2_principal = math.atan2(term_Y_for_atan2, term_X_for_atan2)
    candidates = []
    q2_cand1 = q2_principal
    q3_cand1 = q_sum - q2_cand1
    candidates.append({'q2': q2_cand1, 'q3': q3_cand1, 'type': 'principal'})
    q2_cand2 = q2_principal + 2 * math.pi
    q3_cand2 = q_sum - q2_cand2
    candidates.append({'q2': q2_cand2, 'q3': q3_cand2, 'type': 'plus2pi'})
    q2_cand3 = q2_principal - 2 * math.pi
    q3_cand3 = q_sum - q2_cand3
    candidates.append({'q2': q2_cand3, 'q3': q3_cand3, 'type': 'minus2pi'})
    valid_solutions = []
    for cand in candidates:
        q2_c, q3_c = (cand['q2'], cand['q3'])
        if J_MIN - epsilon_limit <= q2_c <= J_MAX + epsilon_limit and J_MIN - epsilon_limit <= q3_c <= J_MAX + epsilon_limit:
            cand['cost'] = abs(q2_c) + abs(q3_c)
            valid_solutions.append(cand)
    final_q2 = 0.0
    final_q3 = 0.0
    principal_solution_is_valid = False
    for sol in valid_solutions:
        if sol['type'] == 'principal':
            final_q2 = sol['q2']
            final_q3 = sol['q3']
            principal_solution_is_valid = True
            break
    if not principal_solution_is_valid:
        alternatives = [sol for sol in valid_solutions if sol['type'] != 'principal']
        if not alternatives:
            if valid_solutions:
                best_sol = valid_solutions[0]
            else:
                best_sol = candidates[0]
            final_q2 = best_sol['q2']
            final_q3 = best_sol['q3']
        else:
            best_alternative = min(alternatives, key=lambda x: x['cost'])
            final_q2 = best_alternative['q2']
            final_q3 = best_alternative['q3']
    return (q1, final_q2, final_q3)