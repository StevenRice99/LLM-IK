import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Computes the inverse kinematics for a 3-DOF serial manipulator with:
      • Revolute 1 at [0, 0, 0] about Z.
      • Revolute 2 at [0, 0.13585, 0] about Y.
      • Revolute 3 at [0, -0.1197, 0.425] about Y.
      • TCP at [0, 0, 0.39225] (in joint3 frame).

    The overall forward kinematics is:
      TCP = Rz(theta1) * ( Trans(0, 0.13585, 0) *
              [ Ry(theta2) * ( Trans(0, -0.1197, 0.425) *
                              ( Ry(theta3) * Trans(0, 0, 0.39225) ) ]
      )
    Note that in the chain the only “active” rotations occur about Z (joint1)
    and Y (joints 2 and 3). In our convention the target TCP orientation r is given
    in roll–pitch–yaw form. When r[0] is near 0 (no flip), we have:
          yaw  = theta1,
          pitch = theta2+theta3.
    When r[0] is near ±π (a “flipped” configuration) the same physical rotation can be
    represented by
          theta1 = r[2] + π    and    theta2+theta3 = π - r[1],
    which (modulo 2π) is equivalent.
    
    We also note that the constant translation from base to joint2 is [0, 0.13585, 0]
    and from joint2 to joint3 is [0, -0.1197, 0.425]. Since a rotation about Y leaves
    the Y–component unchanged, the effective planar “arm” (joints 2 and 3) sees a fixed
    out‐of–plane offset D = 0.13585 + (–0.1197) = 0.01615. After “undoing” the base 
    rotation by theta1 our TCP must satisfy:
          Rz(–theta1)*p = [ p_planar_x,  D,  p_planar_z ].
    In the joint2 (planar) sub-chain, the known effective link lengths are:
          a = 0.425  (from joint2 to joint3) and 
          b = 0.39225 (from joint3 to TCP).
    In that plane the forward kinematics are:
          p_planar_x = a*sin(theta2) + b*sin(theta2+theta3)
          p_planar_z = a*cos(theta2) + b*cos(theta2+theta3)
    which we invert to get theta2 (and then theta3).

    :param p: target TCP position (x, y, z).
    :param r: target TCP orientation (roll, pitch, yaw) in radians.
              For the non-flipped configuration use r[0] ~= 0, so that
                  theta1 = r[2]    and    theta2+theta3 = r[1].
              For the flipped configuration (r[0] near ±π) use:
                  theta1 = r[2] + π    and    theta2+theta3 = π - r[1].
    :return: (theta1, theta2, theta3) joint angles in radians.
    """
    L1 = 0.13585
    d2 = -0.1197
    a = 0.425
    b = 0.39225
    offset_y = L1 + d2
    x, y, z = p
    R_xy = math.sqrt(x ** 2 + y ** 2)
    if R_xy < 1e-06:
        alpha = 0.0
    else:
        alpha = math.atan2(y, x)
    ratio = offset_y / R_xy if R_xy != 0 else 0.0
    ratio = max(min(ratio, 1.0), -1.0)
    theta1 = alpha - math.asin(ratio)
    tol = 0.001
    if abs(abs(r[0]) - math.pi) < tol:
        theta1_target = r[2] + math.pi
        theta_sum = math.pi - r[1]
    else:
        theta1_target = r[2]
        theta_sum = r[1]
    cos_t1 = math.cos(theta1)
    sin_t1 = math.sin(theta1)
    qx = cos_t1 * x + sin_t1 * y
    qy = -sin_t1 * x + cos_t1 * y
    qz = z
    p_planar_x = qx
    p_planar_z = qz
    term1 = p_planar_x - b * math.sin(theta_sum)
    term2 = p_planar_z - b * math.cos(theta_sum)
    theta2 = math.atan2(term1, term2)
    theta3 = theta_sum - theta2
    return (theta1, theta2, theta3)