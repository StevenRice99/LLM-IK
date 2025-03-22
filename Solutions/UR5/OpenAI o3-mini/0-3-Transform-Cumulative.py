import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Closed‐form inverse kinematics for a 4‐DOF manipulator with:
      - Joint 1: Revolute about Z at [0,0,0]
      - Joint 2: Revolute about Y with translation [0, 0.13585, 0]
      - Joint 3: Revolute about Y with translation [0, -0.1197, 0.425]
      - Joint 4: Revolute about Y with translation [0, 0, 0.39225]
      - TCP offset: [0, 0.093, 0]
    
    Note: Because joints 2–4 rotate about Y and their y–translations sum to
         0.13585 – 0.1197 + 0.093 = 0.10915,
         the TCP’s y–coordinate is fixed at 0.10915 regardless of the arm‐angles.
    
    The IK is decoupled by “removing” the base rotation.
    In particular, if p = (x,y,z) is the target in the world frame and we let
         constant_offset = 0.10915,
    then the base rotation theta1 must satisfy:
         -sin(theta1)*x + cos(theta1)*y = constant_offset.
    Writing this in the form
         cos(theta1 + φ) = constant_offset/√(x²+y²),
    with φ = atan2(x, y), the two algebraic solutions are:
         theta1 = –φ ± arccos(constant_offset/√(x² + y²)).
    Experience with this manipulator indicates that for a given target one branch
    will yield a solution that is consistent with the later positional and orientational
    requirements. We then solve the 2‐R planar chain (with effective link lengths L₁ and L₂)
    in the rotated frame. In that frame (defined by rotating the target by –theta1),
    the x– and z–coordinates satisfy:
         X_target = L₁·sin(theta2) + L₂·sin(theta2+theta3),
         Z_target = L₁·cos(theta2) + L₂·cos(theta2+theta3).
    Finally, since joints 2–4 (all about Y) yield a cumulative rotation
         theta_arm = theta2 + theta3 + theta4,
    we “remove” the base rotation from the desired end–effector orientation (given as roll–pitch–yaw)
    to extract the desired arm rotation. That is, forming
         R' = Rz(–theta1)·R_desired,
    where R_desired = Rz(yaw)*Ry(pitch)*Rx(roll), we set
         theta_total = atan2(R'[0,2], R'[0,0])
    and then
         theta4 = theta_total – (theta2 + theta3).
    
    Because of the dual ambiguities in the solutions for theta1 and theta3, we generate all candidates,
    compute forward kinematics for each, and return the set with minimum position error.
    
    :param p: The target TCP position as (x, y, z)
    :param r: The target TCP roll–pitch–yaw (in radians) as (roll, pitch, yaw)
    :return: A tuple (theta1, theta2, theta3, theta4) of joint values (in radians)
    """
    L1 = 0.425
    L2 = 0.39225
    constant_offset = 0.10915
    x, y, z = p
    roll, pitch, yaw = r
    R_xy = math.sqrt(x ** 2 + y ** 2)
    ratio = constant_offset / R_xy
    ratio = max(min(ratio, 1.0), -1.0)
    delta = math.acos(ratio)
    phi = math.atan2(x, y)
    theta1_candidates = [-phi + delta, -phi - delta]
    cr, sr = (math.cos(roll), math.sin(roll))
    cp, sp = (math.cos(pitch), math.sin(pitch))
    cy, sy = (math.cos(yaw), math.sin(yaw))
    R_des = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    solutions = []
    for theta1 in theta1_candidates:
        c1 = math.cos(theta1)
        s1 = math.sin(theta1)
        Rz_neg = np.array([[c1, s1, 0], [-s1, c1, 0], [0, 0, 1]])
        p_rot_x = c1 * x + s1 * y
        p_rot_y = -s1 * x + c1 * y
        p_rot_z = z
        X_target = p_rot_x
        Z_target = p_rot_z
        r_planar = math.sqrt(X_target ** 2 + Z_target ** 2)
        cos_theta3 = (r_planar ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
        cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
        theta3_options = [math.acos(cos_theta3), -math.acos(cos_theta3)]
        for theta3 in theta3_options:
            gamma = math.atan2(X_target, Z_target)
            theta2 = gamma - math.atan2(L2 * math.sin(theta3), L1 + L2 * math.cos(theta3))
            R_prime = Rz_neg.dot(R_des)
            theta_total = math.atan2(R_prime[0, 2], R_prime[0, 0])
            theta4 = theta_total - (theta2 + theta3)
            candidate = (theta1, theta2, theta3, theta4)
            T1 = np.array([[math.cos(theta1), -math.sin(theta1), 0, 0], [math.sin(theta1), math.cos(theta1), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            T2 = np.array([[math.cos(theta2), 0, math.sin(theta2), 0], [0, 1, 0, 0.13585], [-math.sin(theta2), 0, math.cos(theta2), 0], [0, 0, 0, 1]])
            T3 = np.array([[math.cos(theta3), 0, math.sin(theta3), 0], [0, 1, 0, -0.1197], [-math.sin(theta3), 0, math.cos(theta3), 0.425], [0, 0, 0, 1]])
            T4 = np.array([[math.cos(theta4), 0, math.sin(theta4), 0], [0, 1, 0, 0], [-math.sin(theta4), 0, math.cos(theta4), 0.39225], [0, 0, 0, 1]])
            T_tcp = np.array([[1, 0, 0, 0], [0, 1, 0, 0.093], [0, 0, 1, 0], [0, 0, 0, 1]])
            T_fk = T1.dot(T2).dot(T3).dot(T4).dot(T_tcp)
            pos_fk = T_fk[0:3, 3]
            pos_error = math.sqrt((pos_fk[0] - x) ** 2 + (pos_fk[1] - y) ** 2 + (pos_fk[2] - z) ** 2)
            solutions.append((candidate, pos_error))
    best_candidate, _ = min(solutions, key=lambda s: s[1])
    return best_candidate