import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Closed–form inverse kinematics for the 5–DOF robot.
    
    Robot structure (meters, radians):
      - Revolute 1: translation [0, 0, 0], axis Z.
      - Revolute 2: translation [0, 0.13585, 0], axis Y.
      - Revolute 3: translation [0, -0.1197, 0.425], axis Y.
      - Revolute 4: translation [0, 0, 0.39225], axis Y.
      - Revolute 5: translation [0, 0.093, 0], axis Z.
      - TCP: translation [0, 0, 0.09465].
    
    When joints are zero the forward kinematics yield:
         TCP = (0, 0.10915, 0.9119)
    where the fixed lateral offset in the y–direction is 0.10915 (0.13585 - 0.1197 + 0.093)
    and the TCP z–offset is 0.09465.
    
    We decouple the IK into:
      (a) a rotation q1 about Z that maps a fixed vector [A; offset] to the target’s (x,y)
          components. Here, offset = 0.10915 while A is unknown but constrained by:
              A^2 + offset^2 = x² + y².
          Thus, A = ±sqrt(r_xy² - offset²). We will choose the sign that best 
          yields the correct candidate.
      (b) a planar 2R arm (joints 2 and 3 about Y) that must satisfy:
              L1*sin(q2) + L2*sin(q2+q3) = A   and
              L1*cos(q2) + L2*cos(q2+q3) = B,
          where B = z - tcp_z_offset and the link lengths are:
              L1 = 0.425   and   L2 = 0.39225.
      (c) the redundant joint q4 (about Y) is then chosen to set the overall pitch 
          (q2+q3+q4) to one of two candidates derived from the target’s x–z orientation.
      (d) q5 is not used for positioning and is set to zero.
      
    To resolve redundancy, we generate candidate solutions for:
      - The sign s for A = s*sqrt(r_xy² - offset²) with s in {+1, -1}.
      - Two solutions for q3 (elbow up and down): q3 = ±acos(cos_val)
      - Two choices for the overall pitch “T” for joints 2–4: T = psi and T = psi + π,
        where psi = atan2(x, z).
    For each candidate we compute the (x,y,z) position from:
         A_planar = L1*sin(q2) + L2*sin(q2+q3)
         B_planar = L1*cos(q2) + L2*cos(q2+q3)
         Then, applying q1 (rotation about Z) to the vector [A_planar; offset]:
              FK_x = cos(q1)*A_planar - sin(q1)*offset
              FK_y = sin(q1)*A_planar + cos(q1)*offset
              FK_z = B_planar + tcp_z_offset
    and select the candidate whose forward‐kinematics error is minimal.
    
    :param p: The target TCP position as (x, y, z).
    :return: A tuple (q1, q2, q3, q4, q5) of joint angles (in radians).
    """

    def normalize(angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    offset = 0.10915
    tcp_z_offset = 0.09465
    L1 = 0.425
    L2 = 0.39225
    x, y, z = p
    r_xy = math.sqrt(x * x + y * y)
    if r_xy < offset:
        r_xy = offset
    psi = math.atan2(x, z)
    T_options = [psi, psi + math.pi]
    best_err = float('inf')
    best_solution = None
    for s in [1, -1]:
        A_candidate = s * math.sqrt(max(0, r_xy * r_xy - offset * offset))
        angle_target = math.atan2(y, x)
        angle_ref = math.atan2(offset, A_candidate)
        q1_candidate = normalize(angle_target - angle_ref)
        B = z - tcp_z_offset
        R_eff = math.sqrt(A_candidate * A_candidate + B * B)
        cos_val = (R_eff * R_eff - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
        cos_val = max(-1.0, min(1.0, cos_val))
        for sign_q3 in [1, -1]:
            q3 = sign_q3 * math.acos(cos_val)
            q2 = math.atan2(A_candidate, B) - math.atan2(L2 * math.sin(q3), L1 + L2 * math.cos(q3))
            for T in T_options:
                q4 = normalize(T - (q2 + q3))
                q5 = 0.0
                A_planar = L1 * math.sin(q2) + L2 * math.sin(q2 + q3)
                B_planar = L1 * math.cos(q2) + L2 * math.cos(q2 + q3)
                FK_x = math.cos(q1_candidate) * A_planar - math.sin(q1_candidate) * offset
                FK_y = math.sin(q1_candidate) * A_planar + math.cos(q1_candidate) * offset
                FK_z = B_planar + tcp_z_offset
                err = math.sqrt((FK_x - x) ** 2 + (FK_y - y) ** 2 + (FK_z - z) ** 2)
                if err < best_err:
                    best_err = err
                    best_solution = (normalize(q1_candidate), normalize(q2), normalize(q3), normalize(q4), 0.0)
    if best_solution is None:
        best_solution = (0.0, 0.0, 0.0, 0.0, 0.0)
    return best_solution