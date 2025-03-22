import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Closed-form inverse kinematics solution for the 5-DOF serial manipulator.
    This function computes joint angles (q1, q2, q3, q4, q5) that position the TCP at the target point p = (x, y, z).

    Manipulator structure:
      1. Revolute 1: Origin at [0, 0, 0], rotates about Y.
      2. Fixed translation: [0, -0.1197, 0.425] to Revolute 2.
      3. Revolute 2: Rotates about Y.
      4. Fixed translation: [0, 0, 0.39225] to Revolute 3.
      5. Revolute 3: Rotates about Y.
      6. Fixed translation: [0, 0.093, 0] to Revolute 4.
      7. Revolute 4: Rotates about Z.
      8. Fixed translation: [0, 0, 0.09465] to Revolute 5.
      9. Revolute 5: Rotates about Y.
     10. Fixed translation: [0, 0.0823, 0] to the TCP.

    Notes:
      • With all joint angles zero, the forward kinematics yields:
            TCP = [0, 0.0556, 0.9119].
      • We set q5 = 0 to simplify the redundancy.
      • The TCP vertical coordinate (y) is influenced by the TCP's final offset rotated by q4.
          The relationship is: 0.0823*cos(q4) - 0.0267 = p_y, leading to:
            cos(q4) = (p_y + 0.0267) / 0.0823,
          so that q4 = acos((p_y + 0.0267) / 0.0823).
      • The base joint q1 aligns the manipulator in the x–z plane:
              q1 = atan2(p_x, p_z).
      • In the x–z plane, we remove the fixed offset of 0.425 (Revolute 2 translation) resulting in:
              d_target = sqrt(p_x^2+p_z^2) - 0.425.
      • Define constants:
              A = 0.0823*sin(q4)  (effect from TCP offset rotated by q4)
              B = 0.09465         (translation offset from Revolute 5)
              C = 0.39225         (translation offset from Revolute 3)
          The planar relation is given by:
              d_target^2 = A^2 + B^2 + C^2 + 2*C*(A*sin(q3) + B*cos(q3)).
      • We can express A*sin(q3) + B*cos(q3) as R_ab*sin(q3 + δ), where:
              R_ab = sqrt(A^2+B^2) and δ = atan2(B, A).
          Thus, q3 = asin(K / R_ab) - δ, where K = (d_target^2 - (A^2+B^2+C^2)) / (2*C).
      • Finally, compute q2 to align the effective planar vector:
              Let V_x = -A*cos(q3) + B*sin(q3)
                  and V_z = C + A*sin(q3) + B*cos(q3).
          Then, q2 = atan2(-V_x, V_z).

    Due to possible minor numerical precision issues, the argument (K/R_ab) fed to asin is clamped to [-1, 1].
    Assumes the target is reachable.
    
    :param p: The target TCP position as a tuple (p_x, p_y, p_z).
    :return: A tuple of joint angles (q1, q2, q3, q4, q5) in radians.
    """
    p_x, p_y, p_z = p
    q1 = math.atan2(p_x, p_z)
    cos_q4 = (p_y + 0.0267) / 0.0823
    cos_q4 = max(min(cos_q4, 1.0), -1.0)
    q4 = math.acos(cos_q4)
    q5 = 0.0
    r = math.hypot(p_x, p_z)
    d_target = r - 0.425
    A = 0.0823 * math.sin(q4)
    B = 0.09465
    C = 0.39225
    K = (d_target ** 2 - (A ** 2 + B ** 2 + C ** 2)) / (2 * C)
    R_ab = math.hypot(A, B)
    delta = math.atan2(B, A)
    ratio = K / R_ab
    ratio = max(min(ratio, 1.0), -1.0)
    q3 = math.asin(ratio) - delta
    V_x = -A * math.cos(q3) + B * math.sin(q3)
    V_z = C + A * math.sin(q3) + B * math.cos(q3)
    q2 = math.atan2(-V_x, V_z)
    return (q1, q2, q3, q4, q5)