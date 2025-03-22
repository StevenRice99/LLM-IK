import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Compute a closed‐form inverse kinematics solution for a 4‐DOF serial manipulator.
    
    The robot’s kinematics decouple into position and orientation.
    The position‐only solution uses the same analytic equations as provided:
    
      x² + y² + z² – 0.346395 = 0.3341625·cos(q3)   →  q3 = arccos(clip((x²+y²+z²–0.346395)/0.3341625))
      A = 0.425 + 0.39225·cos(q3)
      B = 0.39225·sin(q3)
      S² = x² + y² – 0.10915²,  S = sqrt(S²) (if nonnegative), and C = z 
      q2 = arctan2(S*A – C*B, S*B + C*A)
      A “helper‐angle” φ = arctan2(0.10915, S) (or π/2 if S is zero)
      An unconstrained solution for the base joint from position is
           q1_pos = arctan2(y, x) – φ
           
    For the full 6‐axis orientation the forward kinematics simplify to:
         R_TCP = Rz(q1)*Ry(q2+q3+q4)
    and a (unique) Euler “rpy” extraction (using the URDF conventions) shows that the 
    overall TCP orientation can be parametrized in one of two equivalent ways:
    
      • Representation “1” (default):
            roll  = 0,
            pitch = q2+q3+q4,
            yaw   = q1.
      • Representation “2” (alternatively):
            roll  = ±π,
            pitch = π − (q2+q3+q4),
            yaw   = q1 − π.
    
    Since the available target orientation r = [roll, pitch, yaw] may be in either form,
    we first “decouple” the problem. We compute the position solution (which yields q2 and q3)
    and then choose one of two candidate solutions so that the orientation is met. In our implementation 
    we “force” the base joint to exactly match the desired yaw (up to its ambiguity) and then choose
    q4 to obtain the required total rotation about Y. Specifically, we define a desired or target 
    base angle q1_target and desired total Y–rotation (theta_total_target) as follows:
    
         If |target_roll| is near 0 (i.e. reachable pose expressed as [0, pitch, yaw]):
              set q1_target = target_yaw      and   theta_total_target = target_pitch.
         Otherwise (if |target_roll| ≈ π):
              set q1_target = target_yaw + π  (normalized to [–π,π]) and 
                  theta_total_target = π − target_pitch.
    
    Note: Because the position–only IK (which computes q1 from the geometry) is ambiguous,
    we compute it (q1_pos) and then decide between the two candidate representations based on 
    which candidate’s base angle is closer (in a modulo‐2π sense) to q1_pos.
    
    The final solution is then:
         q1 = candidate base joint (either q1_target from representation “1” or “2”)
         q2, q3 = as computed from position–IK (they appear only as their sum)
         q4 = theta_total_target − (q2+q3)
    
    This solution guarantees that the forward kinematics give TCP orientation
         R_TCP = Rz(q1)*Ry(q2+q3+q4)
    whose Euler angles (using the same extraction as in the URDF) match r (up to their common ambiguities).
    
    Parameters:
      p: Desired TCP position [x, y, z].
      r: Desired TCP orientation in rpy [roll, pitch, yaw] (radians). Note that r may be given 
         either with roll = 0 or roll = ±π.
    
    Returns:
      A 4-tuple (q1, q2, q3, q4) of joint angles (in radians) that reach the target pose.
    """
    x, y, z = p
    numerator = x ** 2 + y ** 2 + z ** 2 - 0.346395
    denominator = 0.3341625
    cos_q3 = np.clip(numerator / denominator, -1.0, 1.0)
    q3 = np.arccos(cos_q3)
    A = 0.425 + 0.39225 * np.cos(q3)
    B = 0.39225 * np.sin(q3)
    S_sq = x ** 2 + y ** 2 - 0.10915 ** 2
    S = np.sqrt(S_sq) if S_sq >= 0 else 0.0
    C = z
    q2 = np.arctan2(S * A - C * B, S * B + C * A)
    phi = np.arctan2(0.10915, S) if S != 0 else np.pi / 2
    q1_pos = np.arctan2(y, x) - phi
    target_roll, target_pitch, target_yaw = r
    target_roll = (target_roll + np.pi) % (2 * np.pi) - np.pi
    cand1_q1 = target_yaw
    cand1_q4 = target_pitch - (q2 + q3)
    cand2_q1 = (target_yaw + np.pi + np.pi) % (2 * np.pi) - np.pi
    cand2_q4 = np.pi - target_pitch - (q2 + q3)

    def angle_diff(a, b):
        return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)
    if angle_diff(q1_pos, cand1_q1) <= angle_diff(q1_pos, cand2_q1):
        q1 = cand1_q1
        q4 = cand1_q4
    else:
        q1 = cand2_q1
        q4 = cand2_q4
    return (q1, q2, q3, q4)