import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Computes the joint angles (theta1, theta2) for the 2DOF manipulator so that the TCP
    reaches the desired position p = (x, y, z) and orientation r = (roll, pitch, yaw).

    The forward kinematics for the TCP are given by:
        x = 0.425 * sin(theta2) * cos(theta1) - 0.01615 * sin(theta1)
        y = 0.425 * sin(theta2) * sin(theta1) + 0.01615 * cos(theta1)
        z = 0.425 * cos(theta2)

    Because the chain has only two joints, the full (roll, pitch, yaw) pose is overconstrained.
    In fact, for consistency the TCP’s Euler angles (using a convention compatible with the FK)
    must satisfy:
         extracted_pitch = { theta2                      if |theta2| <= π/2,
                             π - theta2                  if theta2 >  π/2,
                            -π - theta2                  if theta2 < -π/2 }
         effective_yaw   = { theta1          if cos(theta2) ≥ 0,
                             theta1 + π      if cos(theta2) <  0  } (wrapped into [–π, π])
    And roll is not directly controlled (it will be 0 or ±π).

    Given that p and r are assumed consistent, we use the following two-step solution:
    
    Step 1: Determine theta2 using the z-coordinate:
      • From z = 0.425*cos(theta2), we have cos(theta2) = z/0.425.
      • A candidate is: candidate = acos(z/0.425), which lies in [0, π].
      • There are two possible joint solutions:
            • If candidate ≤ π/2, then:
                  • Choosing theta2 = candidate yields an extracted pitch of  candidate.
                  • Choosing theta2 = –candidate yields an extracted pitch of –candidate.
            • If candidate > π/2, then:
                  • Choosing theta2 = candidate yields an extracted pitch of π – candidate.
                  • Choosing theta2 = –candidate yields an extracted pitch of –π + candidate.
      • We choose the branch whose extracted pitch is closest to the desired pitch (r[1]).

    Step 2: Determine theta1 using the desired yaw (r[2]):
      • The FK gives effective yaw as:
                  theta1       if cos(theta2) ≥ 0
                  theta1 + π   if cos(theta2) < 0
      • Thus, if cos(theta2) ≥ 0 we set theta1 = r[2],
         and if cos(theta2) < 0 we set theta1 = r[2] – π.
      • Finally, theta1 is wrapped into the interval [–π, π].

    :param p: Desired TCP position (x, y, z)
    :param r: Desired TCP orientation (roll, pitch, yaw) in radians.
              (For this chain, roll is not controlled and may be 0 or ±π.)
    :return: Tuple (theta1, theta2) of joint angles (in radians)
    """
    x, y, z = p
    desired_roll, desired_pitch, desired_yaw = r
    candidate = math.acos(z / 0.425)
    if candidate <= math.pi / 2:
        pos_extracted_pitch = candidate
        neg_extracted_pitch = -candidate
    else:
        pos_extracted_pitch = math.pi - candidate
        neg_extracted_pitch = -math.pi + candidate
    if abs(pos_extracted_pitch - desired_pitch) <= abs(neg_extracted_pitch - desired_pitch):
        theta2 = candidate
    else:
        theta2 = -candidate
    if math.cos(theta2) >= 0:
        theta1 = desired_yaw
    else:
        theta1 = desired_yaw - math.pi
    theta1 = (theta1 + math.pi) % (2 * math.pi) - math.pi
    return (theta1, theta2)