import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Computes a closed‐form analytical inverse kinematics solution for a 5-DOF serial manipulator.
    
    Robot structure (all distances in meters, angles in radians):
      • Revolute Joint 1: at the base, with origin [0, 0, 0] and rotation about Y.
      • Revolute Joint 2: offset by T2 = [0, -0.1197, 0.425] (in its parent frame), rotates about Y.
      • Revolute Joint 3: offset by T3 = [0, 0, 0.39225], rotates about Y.
      • Revolute Joint 4: offset by T4 = [0, 0.093, 0], rotates about Z.
      • Revolute Joint 5: offset by T5 = [0, 0, 0.09465], rotates about Y.
      • TCP: offset by TCP_offset = [0, 0.0823, 0] and a fixed end‐effector yaw rotation psi = 1.570796325.
    
    The forward kinematics for the TCP position are:
      p_TCP = R_y(θ₁)*T2 
              + R_y(θ₁+θ₂)*T3 
              + R_y(θ₁+θ₂+θ₃)*T4 
              + R_y(θ₁+θ₂+θ₃)*R_z(θ₄)*T5 
              + R_y(θ₁+θ₂+θ₃)*R_z(θ₄)*R_y(θ₅)*TCP_offset
              
    Notice that because T2, T3 and T4 have either fixed or Y–only components,
    the x and z coordinates of p_TCP depend only on joints 1–3 (and the fixed T5, which is along z)
    whereas the y component is “lifted” by T2, T4 and the TCP offset rotated by Joint 4.
    
    To decouple the IK:
      1. We first compute the desired end‐effector rotation matrix:
           R_target = R_z(yaw) · R_y(pitch) · R_x(roll)
      2. We extract an “overall arm angle” S ≡ θ₁ + θ₂ + θ₃ from the (0,2) and (2,2) elements of R_target.
         (This is analogous to using the wrist’s approach direction for a spherical wrist.)
      3. Because the translation T5 (of length C_tcp = 0.09465 m) is along the z–axis of the arm,
         we “subtract” its contribution from the target x– and z–coordinates to obtain the effective
         2-link arm position. Specifically, define:
               pos3_x = x_target − C_tcp·sin(S)
               pos3_z = z_target − C_tcp·cos(S)
         Then solve the planar 2-link IK for joints 1 and 2 given link lengths a = 0.425 and
         b = 0.39225. (There are two possible solutions; here we pick the first candidate.)
      4. Then, θ₃ is recovered as:
             θ₃ = S − (θ₁ + θ₂)
      5. The y–coordinate of p_TCP comes out as:
             p_y = T2_y + T4_y + TCP_offset_y·cos(θ₄)
         with T2_y = –0.1197 and T4_y = 0.093 so that T2_y+T4_y = –0.0267.
         Hence, solve for θ₄ via:
             cos(θ₄) = (p_y + 0.0267) / TCP_offset_y,   where TCP_offset_y = 0.0823.
      6. Finally, the orientation from the wrist is given by
             R_total = R_y(S) · R_z(θ₄) · R_y(θ₅) · R_z(psi)
         Equate this with R_target and “isolate” θ₅. In particular, note that
             R_y(θ₅) = [R_y(S)·R_z(θ₄)]ᵀ · R_target · R_z(–psi)
         so that we can extract θ₅ from the (0,0) and (0,2) elements.
    
    Assumptions:
      • The target pose is reachable.
      • When multiple IK solutions exist, the “elbow‐down” (and continuous) candidate is chosen.
    
    :param p: Target TCP position, given as (x, y, z).
    :param r: Target TCP orientation in roll, pitch, yaw (radians).
    :return: Tuple (θ₁, θ₂, θ₃, θ₄, θ₅) of joint angles (in radians).
    """

    def rot_x(a: float) -> np.ndarray:
        ca = math.cos(a)
        sa = math.sin(a)
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    def rot_y(a: float) -> np.ndarray:
        ca = math.cos(a)
        sa = math.sin(a)
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

    def rot_z(a: float) -> np.ndarray:
        ca = math.cos(a)
        sa = math.sin(a)
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    T2 = np.array([0.0, -0.1197, 0.425])
    T3 = np.array([0.0, 0.0, 0.39225])
    T4 = np.array([0.0, 0.093, 0.0])
    T5 = np.array([0.0, 0.0, 0.09465])
    TCP_offset = np.array([0.0, 0.0823, 0.0])
    a = 0.425
    b = 0.39225
    C_tcp = 0.09465
    y_const = T2[1] + T4[1]
    TCP_y = TCP_offset[1]
    psi = 1.570796325
    roll, pitch, yaw = r
    R_target = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    S = math.atan2(R_target[0, 2], R_target[2, 2])
    x_target, y_target, z_target = p
    pos3_x = x_target - C_tcp * math.sin(S)
    pos3_z = z_target - C_tcp * math.cos(S)
    x = pos3_x
    z = pos3_z
    d_sq = x * x + z * z
    cos_theta2 = (d_sq - a * a - b * b) / (2 * a * b)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2_candidate = math.acos(cos_theta2)
    theta2 = -theta2_candidate
    phi = math.atan2(x, z)
    delta = math.atan2(b * math.sin(theta2), a + b * math.cos(theta2))
    theta1 = phi - delta
    theta3 = S - (theta1 + theta2)
    val = (y_target - y_const) / TCP_y
    val = max(min(val, 1.0), -1.0)
    theta4 = math.acos(val)
    R_yS = rot_y(S)
    R_z_theta4 = rot_z(theta4)
    R_temp = R_yS @ R_z_theta4
    M = R_temp.T @ R_target @ rot_z(-psi)
    theta5 = math.atan2(M[0, 2], M[0, 0])
    return (theta1, theta2, theta3, theta4, theta5)