import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Computes a closed‐form analytical inverse kinematics solution for a 5‐DOF manipulator.
    
    The manipulator’s kinematics (all units in meters and angles in radians) are defined by:
      • Revolute 1: origin [0, 0, 0], axis Y.
      • Revolute 2: origin [0, -0.1197, 0.425], axis Y.
      • Revolute 3: origin [0, 0, 0.39225], axis Y.
      • Revolute 4: origin [0, 0.093, 0], axis Z.
      • Revolute 5: origin [0, 0, 0.09465], axis Y.
      • TCP: origin [0, 0.0823, 0] with a fixed rotation Rz(1.570796325).
      
    In this solution the IK is “decoupled” into an arm (joints 1–3) and a wrist (joints 4–5).
    (Note: for a 5‐DOF arm only two independent translational dimensions can be controlled;
     the extra joint is “redundant” and here we assign it to resolve the wrist orientation.)
     
    The solution proceeds as follows:
      1. First the desired TCP rotation R_des is computed from the given rpy (using the
         convention R = Rz(yaw)·Ry(pitch)·Rx(roll)).
      2. A “virtual” arm rotation about Y (which in our chain is the cumulative rotation from joints 1–3)
         is extracted from R_des. Since a pure rotation about Y has the form:
               Ry(ψ) = [[ cosψ, 0, sinψ],
                        [    0, 1,    0],
                        [-sinψ, 0, cosψ]],
         we choose:
               ψ = arctan2( R_des[0,2], R_des[2,2] ).
      3. The wrist “offset” from the TCP is removed. (It is standard in IK to “subtract off” the fixed
         tool transform; here we assume the TCP’s translation is defined in its own (tool) frame.)
         Thus, the wrist center is set to:
               p_wc = p – (R_des · [0, 0.0823, 0])
      4. The y–coordinate of the wrist center is constrained by the fact that joints 2 and 4 contribute a
         fixed vertical offset. In our chain the net offset is:
               y_wrist = (–0.1197 [from link 2] + 0.093 [from link 4]) = –0.0267.
         Then the wrist offset along Y (of magnitude 0.0823) is “resolved” by joint 4 via:
               p_y = (–0.0267) + 0.0823·cos(q4)
         so that
               q4 = arccos((p[1] – (–0.0267)) / 0.0823).
      5. For the arm (joints 1–3) we assume that (by design) only x and z are “assigned” by these joints.
         In our model the forward kinematics of the “arm” (ignoring the wrist’s rotations) yield:
           TCP_x = 0.425·sin(q1) + 0.39225·sin(q1+q2)   and
           TCP_z = 0.425·cos(q1) + 0.39225·cos(q1+q2) + 0.09465.
         We set these equal to the corresponding components of our wrist‐center p_wc.
         (This “planar 2R” IK is computed as follows.)
      6. Define:
             X = p_wc[0]   and   Z = p_wc[2] – 0.09465.
         Then, with
             r = sqrt(X² + Z²),
         the law of cosines gives
             cos(q2) = (r² – 0.425² – 0.39225²) / (2 · 0.425 · 0.39225).
         To be consistent with the “elbow‐down” solution, we choose
             q2 = – arccos( cos(q2) ).
         Also, with
             φ = arctan2(X, Z)
         and
             δ = arctan2( 0.39225·sin(|q2|), 0.425 + 0.39225·cos(|q2|) ),
         we set
             q1 = φ – δ.
      7. The free parameter q3 is then “used up” to achieve the overall arm rotation:
             q3 = ψ – (q1 + q2).
      8. Finally, the wrist orientation is imposed. Noting that the total TCP rotation is given by:
             R_TCP = R_y(ψ) · Rz(q4) · R_y(q5) · Rz(1.570796325)
         we isolate q5 by letting
             A = R_y(ψ) · Rz(q4)
             B = Aᵀ · R_des · Rz(–1.570796325)
         so that B = R_y(q5) and hence
             q5 = arctan2(B[0,2], B[0,0]).
    
    Any solution yielding joint angles (q1, q2, q3, q4, q5) that satisfy these equations is acceptable.
    This implementation (using NumPy) is one acceptable closed‐form solution.
    
    :param p: Desired TCP position as (x, y, z)
    :param r: Desired TCP orientation as (roll, pitch, yaw) [in radians]
    :return: (q1, q2, q3, q4, q5)
    """
    L1 = 0.425
    L2 = 0.39225
    d3 = 0.09465
    tcp_offset = 0.0823
    offset_joint2_y = -0.1197
    offset_joint4_y = 0.093
    wrist_y = offset_joint2_y + offset_joint4_y
    d2 = tcp_offset

    def rot_x(a):
        return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])

    def rot_y(a):
        return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])

    def rot_z(a):
        return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    roll, pitch, yaw = r
    R_des = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    psi = np.arctan2(R_des[0, 2], R_des[2, 2])
    q4 = np.arccos((p[1] - wrist_y) / d2)
    p_wc = np.array(p) - R_des @ np.array([0, tcp_offset, 0])
    X = p_wc[0]
    Z = p_wc[2] - d3
    r_val = np.hypot(X, Z)
    cos_q2 = (r_val ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    q2_candidate = np.arccos(cos_q2)
    q2 = -q2_candidate
    phi = np.arctan2(X, Z)
    delta = np.arctan2(L2 * np.sin(q2_candidate), L1 + L2 * np.cos(q2_candidate))
    q1 = phi - delta
    q3 = psi - (q1 + q2)
    A = rot_y(psi) @ rot_z(q4)
    B = A.T @ R_des @ rot_z(-1.570796325)
    q5 = np.arctan2(B[0, 2], B[0, 0])
    return (q1, q2, q3, q4, q5)