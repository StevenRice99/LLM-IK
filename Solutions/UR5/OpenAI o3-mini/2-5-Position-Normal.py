def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Computes one valid closed‐form inverse kinematics solution for the 4 DOF serial manipulator.
    
    The robot’s geometry (with units in meters and angles in radians) is given by:
      - Joint 1 (Revolute): at [0, 0, 0], rotation about Y.
      - Translation from Joint 1 to Joint 2: (0, 0, L1) with L1 = 0.39225.
      - Joint 2 (Revolute): at [0, 0, L1], rotation about Y.
      - Translation from Joint 2 to Joint 3: (0, L2, 0) with L2 = 0.093.
      - Joint 3 (Revolute): at [0, L2, 0], rotation about Z.
      - Translation from Joint 3 to Joint 4: (0, 0, L3) with L3 = 0.09465.
      - Joint 4 (Revolute): at [0, 0, L3], rotation about Y (redundant for position; set to zero).
      - TCP: offset (0, L4, 0), with L4 = 0.0823.
      
    In our chain the only link that “lifts” the TCP in Y is the combination of the fixed
    offset L2 and the effect of joint 3 via the TCP offset. In fact, one may show that:
    
         p_y = L2 + L4*cos(theta3)
    
    so that theta3 is determined (up to a cosine ambiguity) by:
    
         cos(theta3) = (p_y - L2)/L4.
    
    Because cos(theta3) = cos(theta3 + 2π) the solution for theta3 can be written in two ways.
    Experience with this manipulator (and comparing to a validated IK) shows that the “correct”
    branch for theta3 is not always the one given directly by –acos((p_y – L2)/L4). One can
    obtain the alternative branch by subtracting 2π.
    
    Meanwhile, the X–Z coordinates come from both the “shoulder” joints and the offset produced
    by joint 3 and the TCP. In our derivation the final TCP position is given by:
    
         p_x = L1*sin(theta1) + L3*sin(phi) - L4*sin(theta3)*cos(phi)
         p_y = L2 + L4*cos(theta3)
         p_z = L1*cos(theta1) + L3*cos(phi) + L4*sin(theta3)*sin(phi)
    
    where we have defined:
         phi = theta1 + theta2.
    
    The term L1 comes from the fixed translation from Joint 1 to Joint 2 (along Z in the base),
    and the (phi‐dependent) contributions from joints 2 and 3 appear explicitly.
    
    Our approach is to:
      (a) Solve p_y = L2 + L4*cos(theta3) for theta3. Because cos(theta3) is even this
          yields two candidates:
             candidate 1: theta3 = -acos((p_y - L2) / L4)
             candidate 2: theta3 =  acos((p_y - L2) / L4) - 2π
      (b) For a given choice of theta3 (which fixes B = L4*sin(theta3)), the X–Z equations can be
          rearranged to an equation in the sum phi = theta1 + theta2. In fact one obtains:
    
             (p_x - (A*sin(phi) - B*cos(phi)))² + (p_z - (A*cos(phi) + B*sin(phi)))² = L1²,
    
          where A is L3 and B is L4*sin(theta3). This equation can be rearranged to:
    
             (p_x²+p_z² + A²+B² - L1²)/2 = (p_x*A + p_z*B)*sin(phi) + (p_z*A - p_x*B)*cos(phi).
    
          Writing R0 = sqrt((p_x*A+p_z*B)² + (p_z*A-p_x*B)²) and a phase delta = atan2(p_z*A - p_x*B, 
          p_x*A + p_z*B), we have
             sin(phi + delta) = ((p_x²+p_z² + A²+B² - L1²)/(2*R0)).
    
      (c) This equation (for phi) has two solutions, and for each one theta1 is recovered from:
    
             theta1 = atan2( p_x - (A*sin(phi) - B*cos(phi)),
                             p_z - (A*cos(phi) + B*sin(phi)) )
    
          then theta2 = phi - theta1.
    
      (d) Finally, we “test” the two choices for theta3 and the two corresponding solutions for phi by 
          recomputing the forward position (using the simplified forward equations above) and selecting the 
          candidate that minimizes the error.
    
    Joint 4 is redundant with respect to position so we set theta4 = 0.
    
    This function returns one valid solution (all angles in radians).
    """
    import math
    L1 = 0.39225
    L2 = 0.093
    L3 = 0.09465
    L4 = 0.0823
    p_x, p_y, p_z = p
    cos_val = (p_y - L2) / L4
    cos_val = max(-1.0, min(1.0, cos_val))
    theta3_candidate1 = -math.acos(cos_val)
    theta3_candidate2 = math.acos(cos_val) - 2 * math.pi

    def forward_pos(theta1, theta2, theta3):
        phi = theta1 + theta2
        x_calc = L1 * math.sin(theta1) + L3 * math.sin(phi) - L4 * math.sin(theta3) * math.cos(phi)
        y_calc = L2 + L4 * math.cos(theta3)
        z_calc = L1 * math.cos(theta1) + L3 * math.cos(phi) + L4 * math.sin(theta3) * math.sin(phi)
        return (x_calc, y_calc, z_calc)
    A = L3
    solutions = []
    for theta3 in [theta3_candidate1, theta3_candidate2]:
        B = L4 * math.sin(theta3)
        R0 = math.sqrt((p_x * A + p_z * B) ** 2 + (p_z * A - p_x * B) ** 2)
        if R0 == 0:
            continue
        C = (p_x ** 2 + p_z ** 2 + A ** 2 + B ** 2 - L1 ** 2) / 2.0
        ratio = C / R0
        ratio = max(-1.0, min(1.0, ratio))
        try:
            alpha = math.asin(ratio)
        except Exception:
            alpha = 0.0
        delta = math.atan2(p_z * A - p_x * B, p_x * A + p_z * B)
        phi_candidates = [alpha - delta, math.pi - alpha - delta]
        for phi in phi_candidates:
            num = p_x - (A * math.sin(phi) - B * math.cos(phi))
            den = p_z - (A * math.cos(phi) + B * math.sin(phi))
            theta1_candidate = math.atan2(num, den)
            theta2_candidate = phi - theta1_candidate
            x_calc, y_calc, z_calc = forward_pos(theta1_candidate, theta2_candidate, theta3)
            error = math.sqrt((x_calc - p_x) ** 2 + (y_calc - p_y) ** 2 + (z_calc - p_z) ** 2)
            solutions.append((error, theta1_candidate, theta2_candidate, theta3, 0.0))
    if solutions:
        best = min(solutions, key=lambda s: s[0])
        return (best[1], best[2], best[3], best[4])
    else:
        return (0.0, 0.0, 0.0, 0.0)