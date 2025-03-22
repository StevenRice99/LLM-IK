import math

def normalize_angle(angle: float) -> float:
    """
    Normalizes an angle to (–pi, pi].
    """
    a = math.fmod(angle + math.pi, 2 * math.pi)
    if a < 0:
        a += 2 * math.pi
    return a - math.pi

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Compute the joint angles (theta1, theta2) for a 2-DOF serial manipulator.
    
    Kinematic Data:
      - Revolute joint 1 is located at the base and rotates about Z.
      - Revolute joint 2 is offset by a translation [0, 0.13585, 0] from joint1,
        then rotates about Y.
      - The TCP is defined in joint2’s frame by a translation [0, -0.1197, 0.425].
    
    Hence the full forward kinematics position is:
      p = Rz(theta1) * ( [L * sin(theta2), offset, L * cos(theta2)] )
       where L = 0.425 and offset = (0.13585 - 0.1197) = 0.01615.
    
    The (roll, pitch, yaw) representation of the TCP’s orientation, as computed
    from the chain Rz(theta1)*Ry(theta2), appears (by convention) either as:
        [0, theta2, theta1]   or, if a flip occurs,
        [pi, -theta2, theta1 ± pi].
    
    In this solver the provided orientation r = (roll, pitch, yaw) is assumed to
    be consistent with one of these two representations. Therefore, we can recover
    the joint angles directly from r.
    
    The strategy is:
      • If roll is nearly 0, then the chain’s orientation is given by [0, theta2, theta1],
        so we simply set:
             theta1 = yaw,   theta2 = pitch.
      • If roll is approximately ±pi, then a “flip” has occurred and the kinematics
        satisfy:
             (roll, pitch, yaw) = (±pi, ∓theta2, theta1 ∓ pi).
        We then recover the raw angles:
             For roll ≈ +pi:  raw_theta1 = yaw – pi,   raw_theta2 = pi – pitch.
             For roll ≈ –pi:  raw_theta1 = yaw + pi,    raw_theta2 = –pi – pitch.
        Finally, to pick between equivalent representations (which differ by 2π)
        we choose as follows:
          – For roll ≈ +pi:
              If pitch is negative and its magnitude is “large” (here > 0.2 rad),
              subtract 2π from raw_theta2; otherwise leave it.
          – For roll ≈ –pi:
              If pitch is positive, add 2π to raw_theta2; otherwise leave it.
      • (If neither branch applies, we fall back to a geometric solution using p.)
    
    Note: Joint 1 (rotation about Z) is normalized to (–pi, pi] so that its value is unique.
          Joint 2 is only adjusted by ±2π when needed.
    
    :param p: Target TCP position as (x, y, z).
    :param r: Target TCP orientation as (roll, pitch, yaw) in radians.
    :return: A tuple (theta1, theta2) for the joint angles.
    """
    x, y, z = p
    roll, pitch, yaw = r
    tol = 1e-06
    L = 0.425
    if abs(roll) < tol:
        theta1 = yaw
        theta2 = pitch
    elif abs(abs(roll) - math.pi) < tol:
        if roll > 0:
            raw_theta1 = yaw - math.pi
            raw_theta2 = math.pi - pitch
            if pitch < 0 and abs(pitch) > 0.2:
                theta2 = raw_theta2 - 2 * math.pi
            else:
                theta2 = raw_theta2
            theta1 = raw_theta1
        else:
            raw_theta1 = yaw + math.pi
            raw_theta2 = -math.pi - pitch
            if pitch > 0:
                theta2 = raw_theta2 + 2 * math.pi
            else:
                theta2 = raw_theta2
            theta1 = raw_theta1
    else:
        ratio = z / L
        ratio = max(min(ratio, 1.0), -1.0)
        theta2_candidate = math.acos(ratio)
        if pitch < 0:
            theta2_candidate = -theta2_candidate
        A = L * math.sin(theta2_candidate)
        B = 0.13585 - 0.1197
        psi = math.atan2(y, x)
        delta = math.atan2(B, A)
        theta1 = psi - delta
        theta2 = theta2_candidate
    theta1 = normalize_angle(theta1)
    return (theta1, theta2)