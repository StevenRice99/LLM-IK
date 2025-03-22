import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    CLOSED-FORM INVERSE KINEMATICS FOR A 2-DOF ROBOT

    Robot structure (from DETAILS):
      1) Revolute joint #1 about Z-axis at base (position [0, 0, 0]).
         Offset to the next joint is [0, 0.13585, 0].
      2) Revolute joint #2 about Y-axis.
         Final link (TCP) offset from joint #2 is [0, -0.1197, 0.425].

    We want to find:
        q1 (joint1 angle about Z), 
        q2 (joint2 angle about Y)
    such that:
        position p = [px, py, pz] is reached,
        orientation r = [roll, pitch, yaw] is approached as closely as possible
        by the rotation Rz(q1)*Ry(q2).  

    Because the same rotation can often be represented by multiple (q1, q2)
    pairs for a 2-DOF chain, we specifically choose the pair that yields the 
    smallest sum-of-squared angle differences versus the target (roll, pitch, yaw),
    handling 2π wrapping. This helps ensure sign alignment with the user request 
    (e.g. picking q1 ~ -π vs. +π if that best matches the target yaw).

    :param p: (px, py, pz), desired TCP position in world coords
    :param r: (roll, pitch, yaw), desired TCP orientation in radians
    :return: (q1, q2) in radians
    """
    px, py, pz = p
    roll_tgt, pitch_tgt, yaw_tgt = r
    offset_y = 0.01615
    link_z = 0.425

    def clamp(value, minv, maxv):
        return max(minv, min(value, maxv))

    def angle_diff(a, b):
        """
        Signed difference in [-pi, pi], for angles a, b (radians).
        """
        d = (a - b) % (2 * math.pi)
        if d > math.pi:
            d -= 2 * math.pi
        return d

    def orientation_error(r_actual, r_target):
        """
        Sum of squared angle differences (roll, pitch, yaw),
        each difference in [-pi, pi].
        """
        return sum((angle_diff(a, b) ** 2 for a, b in zip(r_actual, r_target)))

    def build_RzRy(q1, q2):
        """
        Rotation matrix for Rz(q1)*Ry(q2).
        """
        c1, s1 = (math.cos(q1), math.sin(q1))
        c2, s2 = (math.cos(q2), math.sin(q2))
        return np.array([[c1 * c2, -s1, c1 * s2], [s1 * c2, c1, s1 * s2], [-s2, 0.0, c2]], dtype=float)

    def extract_rpy_urdf(R):
        """
        Extract (roll, pitch, yaw) from R = Rz(yaw)*Ry(pitch)*Rx(roll),
        using standard URDF conventions:
          roll  = atan2(R[2,1], R[2,2])
          pitch = -asin(R[2,0])
          yaw   = atan2(R[1,0], R[0,0])
        """
        r_x = math.atan2(R[2, 1], R[2, 2])
        sy = -R[2, 0]
        sy = clamp(sy, -1.0, 1.0)
        r_y = math.asin(sy)
        r_z = math.atan2(R[1, 0], R[0, 0])
        return (r_x, r_y, r_z)

    def solve_q1(px, py, offy, s2):
        """
        Solve from:
          px = -offy*sin(q1) + (0.425)*cos(q1)*s2
          py =  offy*cos(q1) + (0.425)*sin(q1)*s2
        => single solution for q1 in [-pi, pi).
        """
        d = 0.425 * s2
        c = offy
        denom = d * d + c * c
        if abs(denom) < 1e-15:
            return None
        X = (d * px + c * py) / denom
        Y = (-c * px + d * py) / denom
        return math.atan2(Y, X)
    c2 = clamp(pz / link_z, -1.0, 1.0)
    q2a = math.acos(c2)
    q2b = -q2a
    candidates = []
    for q2cand in [q2a, q2b]:
        s2cand = math.sin(q2cand)
        q1cand = solve_q1(px, py, offset_y, s2cand)
        if q1cand is not None:
            candidates.append((q1cand, q2cand))
    if not candidates:
        return (0.0, 0.0)
    best_sol = None
    best_err = float('inf')
    for q1c, q2c in candidates:
        R_c = build_RzRy(q1c, q2c)
        rpy_c = extract_rpy_urdf(R_c)
        err = orientation_error(rpy_c, (roll_tgt, pitch_tgt, yaw_tgt))
        if err < best_err:
            best_err = err
            best_sol = (q1c, q2c)
    if best_sol is None:
        return (0.0, 0.0)

    def wrap_pm_pi(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi
    q1_final = wrap_pm_pi(best_sol[0])
    q2_final = wrap_pm_pi(best_sol[1])
    two_pi = 2 * math.pi

    def clamp_limits(a):
        while a > 6.2831853:
            a -= two_pi
        while a < -6.2831853:
            a += two_pi
        return a
    q1_final = clamp_limits(q1_final)
    q2_final = clamp_limits(q2_final)
    return (q1_final, q2_final)