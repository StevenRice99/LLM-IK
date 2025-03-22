def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Computes a closed‐form inverse kinematics solution for a 6 DOF serial manipulator.
    
    The manipulator has the following structure:
      - Joint 1 rotates about Z at the base.
      - Joints 2 and 3 (both about Y) form a planar 2-link “arm”.
          • Link 2 is offset from the base by [0, 0.13585, 0].
          • The “shoulder” for the planar solution is effectively at
            [0, (0.13585 - 0.1197), 0] = [0, 0.01615, 0] in the base frame.
          • The two links (in the shoulder’s x–z plane) have lengths:
                A = 0.425 (from shoulder to elbow)
                B = 0.39225 (from elbow to wrist center)
      - Joints 4, 5, and 6 form a spherical wrist.
        The TCP offset relative to joint 6 is defined by:
          • Translation: [0, 0.0823, 0]
          • Fixed rotation (rpy): [0, 0, 1.570796325]
        Based on forward kinematics analysis, the effective TCP offset in joint6’s frame 
        is taken as d_tcp = [0, 0, 0.0823].
    
    The procedure is:
      1. Compute the wrist center (p_wc) by subtracting the TCP offset (expressed in the target frame)
         from the target TCP position.
      2. Determine q1 (rotation about Z) by requiring that, when p_wc is rotated into the frame of joint2,
         its offset along the Y–axis equals the known shoulder offset (d_shoulder = 0.13585 – 0.1197 = 0.01615).
      3. Transform p_wc into the shoulder frame and solve the 2‐link planar problem (in the x–z plane)
         to determine q2 and q3. In this formulation the “shoulder” is at [0, d_shoulder, 0] in joint2’s frame.
         Note: There are two possible solutions (elbow “up” and “down”). Here we choose a branch based on 
         the sign of the x–coordinate in the shoulder frame:
           • If x >= 0: use q3 = π – temp and q2 = φ – β.
           • If x < 0:  use q3 = temp – π and q2 = φ + β.
         where 
             D = sqrt(x²+z²) is the planar distance from the shoulder to the wrist center,
             temp = acos( clamp((D² - A² - B²) / (2*A*B) ) ),
             φ = atan2(x, z),
             and β = atan2( B*sin(temp), A - B*cos(temp) ).
      4. Compute R_0_3 = Rz(q1) · Ry(q2+q3).
      5. Find the wrist rotation matrix R_w from:
             R_w = (R_0_3)ᵀ · R_target · (R_tcp_fixed)ᵀ,
         where R_target is built from the target Euler angles (roll, pitch, yaw) using 
         the convention R_target = Rz(yaw) · Ry(pitch) · Rx(roll) and 
         R_tcp_fixed = Rz(1.570796325).
      6. Finally, extract the wrist angles q4, q5, and q6 from R_w using:
             q5 = atan2( sqrt(R_w[0,1]² + R_w[2,1]²), R_w[1,1] )
             q6 = atan2( R_w[1,2], R_w[1,0] )
             q4 = atan2( R_w[2,1], -R_w[0,1] )
    
    Returns:
      A tuple (q1, q2, q3, q4, q5, q6) of the joint angles (in radians).
    """
    import numpy as np
    from math import sin, cos, atan2, acos, asin, sqrt, pi

    def R_x(angle: float) -> np.ndarray:
        return np.array([[1, 0, 0], [0, cos(angle), -sin(angle)], [0, sin(angle), cos(angle)]])

    def R_y(angle: float) -> np.ndarray:
        return np.array([[cos(angle), 0, sin(angle)], [0, 1, 0], [-sin(angle), 0, cos(angle)]])

    def R_z(angle: float) -> np.ndarray:
        return np.array([[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]])
    roll, pitch, yaw = r
    R_target = R_z(yaw) @ R_y(pitch) @ R_x(roll)
    d2 = 0.13585
    offset_y_joint23 = -0.1197
    d_shoulder = d2 + offset_y_joint23
    A = 0.425
    B = 0.39225
    d_tcp = np.array([0, 0, 0.0823])
    R_tcp_fixed = R_z(1.570796325)
    p_vec = np.array(p)
    p_wc = p_vec - R_target @ d_tcp
    Xw, Yw, Zw = p_wc
    R_xy = sqrt(Xw ** 2 + Yw ** 2)
    phi_wc = atan2(Yw, Xw)
    ratio = d_shoulder / R_xy if R_xy != 0 else 0.0
    if ratio > 1.0:
        ratio = 1.0
    elif ratio < -1.0:
        ratio = -1.0
    q1 = phi_wc - asin(ratio)
    p_wc_shoulder = R_z(-q1) @ p_wc - np.array([0, d_shoulder, 0])
    x_p = p_wc_shoulder[0]
    z_p = p_wc_shoulder[2]
    phi = atan2(x_p, z_p)
    D = sqrt(x_p ** 2 + z_p ** 2)
    num = D ** 2 - A ** 2 - B ** 2
    den = 2 * A * B
    cos_arg = num / den if den != 0 else 0.0
    if cos_arg > 1.0:
        cos_arg = 1.0
    elif cos_arg < -1.0:
        cos_arg = -1.0
    temp = acos(cos_arg)
    beta = atan2(B * sin(temp), A - B * cos(temp))
    if x_p >= 0:
        q3 = pi - temp
        q2 = phi - beta
    else:
        q3 = temp - pi
        q2 = phi + beta
    R_0_3 = R_z(q1) @ R_y(q2 + q3)
    R_w = R_0_3.T @ R_target @ R_tcp_fixed.T
    s5 = sqrt(R_w[0, 1] ** 2 + R_w[2, 1] ** 2)
    q5 = atan2(s5, R_w[1, 1])
    q6 = atan2(R_w[1, 2], R_w[1, 0])
    q4 = atan2(R_w[2, 1], -R_w[0, 1])
    return (q1, q2, q3, q4, q5, q6)