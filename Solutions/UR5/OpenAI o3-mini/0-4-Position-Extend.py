import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Computes a closed‐form analytical solution for the 5–DOF serial manipulator
    so as to reach a desired TCP position p = [x, y, z].
    
    The robot’s geometry is:
      • Joint 1 (Revolute): origin [0, 0, 0], axis Z.
      • Joint 2 (Revolute): origin [0, 0.13585, 0], axis Y.
      • Joint 3 (Revolute): origin [0, -0.1197, 0.425], axis Y.
      • Joint 4 (Revolute): origin [0, 0, 0.39225], axis Y.
      • Joint 5 (Revolute): origin [0, 0.093, 0],   axis Z.
      • TCP: [0, 0, 0.09465] (expressed in link‑5’s coordinates).
    
    (Note: An “effective” offset along the y–axis of the arm,
           d = 0.10915, is obtained from 0.13585 + (–0.1197) + 0.093.)
    
    Because the forward kinematics (when all joints are zero) give:
         TCP = [0,    0.10915, 0.9119],
    the “wrist” (origin of link\u202f5) is at
         p_w0 = [0,    0.10915, 0.9119 – 0.09465] = [0, 0.10915, 0.81725].
    
    One standard way to “decouple” position from orientation is to first solve an inverse kinematics
    (IK) problem for the arm sub–chain (joints 1–4) to position the wrist at a target p_w.
    Here we assume that by “fixing” the redundant joint (joint\u202f5) at zero the TCP offset appears
    as an additive translation along the base–z direction. Thus we choose:
         p_w = p – [0, 0, tcp_offset]
    and then solve for q1, q2, and q3 from
         ||p_w||² – (L1² + L2² + d²) = 2 L1 L2 cos(q3)
    where L1 = 0.425 and L2 = 0.39225.
    
    In many closed–form solutions the correct (elbow–down) branch is obtained by “flipping”
    the result from the arccos. Here we adopt the following prescriptions:
    
      1. Compute p_w = p – [0, 0, 0.09465].
      2. Compute q3 = – arccos( clip( (x²+y²+z² – (L1²+L2²+d²))/(2 L1 L2) , –1, 1) ).
      3. Define
             A = L1 + L2 cos(q3)
             B = L2 sin(q3)
      4. In the “shoulder‐plane” the auxiliary variable S is given by:
             S = sqrt( x²+y² – d² )
         and then
             q2 = arctan2( S A – z B, S B + z A )
         with an “offset angle” φ = arctan2(d, S) so that
             q1 = arctan2(y,x) + φ – π.
      5. Finally, use the arm redundancy (which does not affect TCP position)
         to “flatten” the wrist by selecting
             q4 = – (q2 + q3)
         and choose q5 = 0.
    
    This solution yields one acceptable branch of the infinite set of valid IK solutions.
    (Because the 5–DOF position–only IK is under–constrained, many other solutions exist.)
    
    :param p: Desired TCP position [x, y, z] in meters.
    :return: A 5–tuple (q1, q2, q3, q4, q5) in radians.
    """
    tcp_offset = 0.09465
    d = 0.10915
    L1 = 0.425
    L2 = 0.39225
    x_tcp, y_tcp, z_tcp = p
    p_w = np.array([x_tcp, y_tcp, z_tcp]) - np.array([0, 0, tcp_offset])
    x, y, z = p_w
    const_sum = L1 ** 2 + L2 ** 2 + d ** 2
    numerator = x ** 2 + y ** 2 + z ** 2 - const_sum
    denominator = 2 * L1 * L2
    cos_q3 = np.clip(numerator / denominator, -1.0, 1.0)
    q3 = -np.arccos(cos_q3)
    A = L1 + L2 * np.cos(q3)
    B = L2 * np.sin(q3)
    S_sq = x ** 2 + y ** 2 - d ** 2
    S = np.sqrt(S_sq) if S_sq > 0 else 0.0
    C = z
    q2 = np.arctan2(S * A - C * B, S * B + C * A)
    phi = np.arctan2(d, S) if S != 0 else np.pi / 2
    q1 = np.arctan2(y, x) + phi - np.pi
    q4 = -(q2 + q3)
    q5 = 0.0
    return (q1, q2, q3, q4, q5)