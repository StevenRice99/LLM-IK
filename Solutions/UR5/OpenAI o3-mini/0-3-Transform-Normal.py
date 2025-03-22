def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Computes the joint angles (theta1, theta2, theta3, theta4) required for the 4-DOF serial manipulator
    to attain a desired TCP position "p" and orientation "r".
    
    Manipulator details (all units in meters, angles in radians):
      • Revolute 1: translation = [0, 0, 0], rotation about Z (joint angle = theta1)
      • Revolute 2: translation = [0, 0.13585, 0], rotation about Y (joint angle = theta2)
      • Revolute 3: translation = [0, -0.1197, 0.425], rotation about Y (joint angle = theta3)
      • Revolute 4: translation = [0, 0, 0.39225], rotation about Y (joint angle = theta4)
      • TCP: translation = [0, 0.093, 0]
    
    Kinematics chain:
      T0_TCP = Rz(theta1) · Trans(0, 0.13585, 0) · Ry(theta2) · Trans(0, -0.1197, 0.425) · 
               Ry(theta3) · Trans(0, 0, 0.39225) · Ry(theta4) · Trans(0, 0.093, 0)
    
    Notes on decoupling:
      - The first joint (theta1) rotates about Z. We set theta1 equal to the desired yaw (r[2]).
      - The next three joints rotate about Y so their net effect is a rotation about Y. We set:
            theta_total = theta2 + theta3 + theta4 = desired pitch (r[1])
      - The TCP has an additional fixed translation along Y (0.093 m) in the local frame of joint 4.
    
    Steps:
      1. Set theta1 = r[2] (desired yaw).
      2. Set theta_total = r[1] (desired pitch). Then, later, compute theta4 = theta_total - (theta2 + theta3).
      3. Compute the wrist center position p_wrist in world coordinates, accounting for the TCP offset.
           p_wrist = p - (Rz(theta1) · [0, 0.093, 0])
         where Rz(theta1)*[0, 0.093, 0] = [ -sin(theta1)*0.093, cos(theta1)*0.093, 0 ]
      4. Transform p_wrist into the coordinate frame of joint 2. Joint 2's origin in the world is given by Rz(theta1) · [0, 0.13585, 0].
         Using the inverse rotation (Rz(-theta1)), we compute:
           x_joint2 = cos(theta1)*p_wrist_x + sin(theta1)*p_wrist_y
           y_joint2 = -sin(theta1)*p_wrist_x + cos(theta1)*p_wrist_y
         and subtract the translation [0, 0.13585, 0]:
           p_w2 = (x_joint2, y_joint2 - 0.13585, p_wrist_z)
         (In an ideal scenario, the y-component of p_w2 should be -0.1197.)
      5. For the planar 2R problem, let:
           X = p_w2_x and Z = p_w2_z
         With effective link lengths:
           L1 = 0.425   (from joint 2 to joint 3 along Z)
           L2 = 0.39225 (from joint 3 to joint 4 along Z)
      6. Solve for theta3 using the law of cosines:
             cos(theta3) = (X^2 + Z^2 - L1^2 - L2^2) / (2 * L1 * L2)
         Then choose the elbow “up” solution:
             theta3 = atan2( sqrt(1 - cos(theta3)^2), cos(theta3) )
      7. Solve for theta2:
             theta2 = atan2(X, Z) - atan2(L2*sin(theta3), L1 + L2*cos(theta3))
      8. Finally, recover theta4 so that theta2 + theta3 + theta4 = theta_total:
             theta4 = theta_total - (theta2 + theta3)
    
    :param p: Desired TCP position (x, y, z) in world coordinates.
    :param r: Desired TCP orientation (roll, pitch, yaw) in radians.
    :return: A 4-tuple (theta1, theta2, theta3, theta4) representing the joint angles.
    """
    import math
    d2 = 0.13585
    d_TCP = 0.093
    d3_y = 0.1197
    d3_z = 0.425
    d4 = 0.39225
    theta1 = r[2]
    theta_total = r[1]
    sin1 = math.sin(theta1)
    cos1 = math.cos(theta1)
    p_wrist = (p[0] + sin1 * d_TCP, p[1] - cos1 * d_TCP, p[2])
    x_joint2 = cos1 * p_wrist[0] + sin1 * p_wrist[1]
    y_joint2 = -sin1 * p_wrist[0] + cos1 * p_wrist[1]
    z_joint2 = p_wrist[2]
    p_w2 = (x_joint2, y_joint2 - d2, z_joint2)
    X = p_w2[0]
    Z = p_w2[2]
    L1 = d3_z
    L2 = d4
    r_sq = X ** 2 + Z ** 2
    cos_theta3 = (r_sq - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = math.atan2(math.sqrt(1 - cos_theta3 ** 2), cos_theta3)
    theta2 = math.atan2(X, Z) - math.atan2(L2 * math.sin(theta3), L1 + L2 * math.cos(theta3))
    theta4 = theta_total - (theta2 + theta3)
    return (theta1, theta2, theta3, theta4)