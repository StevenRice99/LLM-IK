def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    A simplified closed-form, position-only inverse kinematics solution for this 5-DOF arm.
    -------------------------------------------------------------------------
    Given the link geometry, we will:
      1) Compute joint1 (q1) = atan2(y, x) to point the arm in the plane of (x, y).
      2) Approximate the remaining links as a simplified 2-link planar mechanism
         (joints q2 and q3 about Y) reaching out in the plane formed by the radial
         distance from the robot axis (r = sqrt(x^2 + y^2)) and the vertical z-axis.
         We treat Link-2 as length L1 = 0.425, Link-3 as length L2 = 0.39225.
      3) Combine the small end-link offsets (0.093 + 0.09465 = 0.18765) into a
         "wrist" offset we subtract from the radial distance so that typical 2-link
         geometry formulas apply. Then we solve for q2, q3 using the law of cosines.
      4) We set joint4 = 0 and joint5 = 0 for a default orientation about those axes.
    
    Because the actual URDF includes small translations (0.13585, -0.1197, etc.), this
    simplified closed-form version may not be perfectly exact for every (x, y, z),
    but it provides an algebraic (non-iterative) solution that will run quickly.
    We assume all requested positions are within the manipulator’s reachable workspace.
    """
    import math
    x, y, z = p
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        q1 = 0.0
    else:
        q1 = math.atan2(y, x)
    r = math.sqrt(x * x + y * y)
    R = math.sqrt(r * r + z * z)
    wrist_offset = 0.093 + 0.09465
    R_eff = max(0.0, R - wrist_offset)
    L1 = 0.425
    L2 = 0.39225
    if abs(R_eff) < 1e-12:
        q2 = math.pi / 2
        q3 = 0.0
    else:
        cos_elbow = (L1 * L1 + L2 * L2 - R_eff * R_eff) / (2 * L1 * L2)
        cos_elbow = max(-1.0, min(1.0, cos_elbow))
        elbow_angle = math.acos(cos_elbow)
        q3 = math.pi - elbow_angle
        cos_alpha = (R_eff * R_eff + L1 * L1 - L2 * L2) / (2 * L1 * R_eff)
        cos_alpha = max(-1.0, min(1.0, cos_alpha))
        alpha = math.acos(cos_alpha)
        phi = math.atan2(z, r)
        q2 = phi + alpha
    q4 = 0.0
    q5 = 0.0
    return (q1, q2, q3, q4, q5)