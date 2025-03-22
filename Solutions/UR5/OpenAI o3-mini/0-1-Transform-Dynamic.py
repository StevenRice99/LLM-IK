import math

def normalize_angle(angle: float) -> float:
    """Normalizes an angle to the range (-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle <= -math.pi:
        angle += 2 * math.pi
    return angle

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Inverse kinematics for a 2-DoF manipulator with:
      - Joint 1: revolute about Z at [0, 0, 0]
      - Joint 2: revolute about Y at [0, 0.13585, 0]
      - TCP offset (in Joint 2 frame): [0, -0.1197, 0.425]
    
    The forward kinematics (position only) work out to:
      p_x = cos(theta1) * (0.425*sin(theta2)) - sin(theta1) * (0.13585 - 0.1197)
      p_y = sin(theta1) * (0.425*sin(theta2)) + cos(theta1) * (0.13585 - 0.1197)
      p_z = 0.425 * cos(theta2)
      
    Since the chain has only 2 DOF the desired TCP orientation is not fully free.
    We make use of the extra information provided in r (its yaw component) to resolve
    the branch ambiguity. In our convention the TCP’s yaw is the orientation about Z
    of the composite rotation Rz(theta1)*Ry(theta2). (Note that when cos(theta2) < 0,
    the yaw effectively shifts by pi.)
    
    The method is as follows:
      1. From p_z = 0.425*cos(theta2) we get cos(theta2)=p_z/0.425.
         Two possible solutions exist: theta2 = +acos(p_z/0.425) or theta2 = -acos(p_z/0.425).
      2. In the xy‐plane the position satisfies:
             [p_x, p_y] = Rz(theta1) * [0.425*sin(theta2); (0.13585 - 0.1197)]
         so that
             theta1 = atan2(p_y, p_x) – atan2(0.01615, 0.425*sin(theta2))
         where 0.01615 = 0.13585 – 0.1197.
      3. We then obtain two candidate solutions:
             Candidate 1: (theta1₁, theta2₁) with theta2₁ = acos(p_z/0.425)
             Candidate 2: (theta1₂, theta2₂) with theta2₂ = -acos(p_z/0.425)
      4. Although both solutions yield the same TCP position, they lead to different
         TCP orientations. In our manipulator the TCP’s yaw (extracted from R = Rz(theta1)*Ry(theta2))
         will equal theta1 when cos(theta2) >= 0 but (theta1 + pi) when cos(theta2) < 0.
      5. We compute the candidate yaw for each solution and then pick the candidate 
         whose yaw is nearest (modulo 2pi) to the target yaw from r (i.e. r[2]).
    
    :param p: Target TCP position as (x, y, z).
    :param r: Target TCP orientation (roll, pitch, yaw) in radians. Although the orientation
              is not independently controllable, we use its yaw component r[2] to resolve branch ambiguity.
    :return: A tuple (theta1, theta2) with the joint angles in radians.
    """
    x, y, z = p
    D = 0.13585 - 0.1197
    cos_theta2 = z / 0.425
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2_candidate1 = math.acos(cos_theta2)
    theta2_candidate2 = -math.acos(cos_theta2)

    def compute_theta1(theta2: float) -> float:
        return math.atan2(y, x) - math.atan2(D, 0.425 * math.sin(theta2))
    theta1_candidate1 = compute_theta1(theta2_candidate1)
    theta1_candidate2 = compute_theta1(theta2_candidate2)

    def compute_yaw(theta1: float, theta2: float) -> float:
        if math.cos(theta2) >= 0:
            yaw = theta1
        else:
            yaw = theta1 + math.pi
        return normalize_angle(yaw)
    yaw_candidate1 = compute_yaw(theta1_candidate1, theta2_candidate1)
    yaw_candidate2 = compute_yaw(theta1_candidate2, theta2_candidate2)
    target_yaw = normalize_angle(r[2])
    diff1 = abs(normalize_angle(yaw_candidate1 - target_yaw))
    diff2 = abs(normalize_angle(yaw_candidate2 - target_yaw))
    if diff1 <= diff2:
        chosen_theta1 = theta1_candidate1
        chosen_theta2 = theta2_candidate1
    else:
        chosen_theta1 = theta1_candidate2
        chosen_theta2 = theta2_candidate2
    return (chosen_theta1, chosen_theta2)