import math


def inverse_kinematics(p: list) -> list:
    """
    Solve the inverse kinematics to reach the position in the reference frame of the chain's origin.

    :param p: The position for the end effector to reach in Cartesian [x, y, z].
    :return: A list of the joint values in radians [theta1, theta2, theta3] needed for the end effector to reach the position.
    :raises ValueError: If the target position is out of the robot's reachable workspace.
    """
    x, y, z = p

    # Compute theta1 correctly
    theta1 = math.atan2(x, y)

    # Compute the projection in the XY-plane
    A = math.sqrt(x ** 2 + y ** 2)

    # Compute K and M based on the FK equations
    # z = 0.4 + 0.4*cos(theta2) + 0.8*cos(theta2 + theta3)
    # A = 0.4*(sin(theta2) + sin(theta2 + theta3))
    # Thus,
    # K = (A) / 0.4 = sin(theta2) + sin(theta2 + theta3)
    # M = (z - 0.4) / 0.8 = cos(theta2) + cos(theta2 + theta3)
    K = A / 0.4
    M = (z - 0.4) / 0.8

    # Numerical precision tolerance
    epsilon = 1e-6

    # Check if M and K are within valid ranges based on trigonometric identities
    # |sin(theta2) + sin(theta2 + theta3)| <= 2
    # |cos(theta2) + cos(theta2 + theta3)| <= 2
    if abs(K) > 2 + epsilon:
        raise ValueError("No solution: K is out of valid range (-2 <= K <= 2).")
    if abs(M) > 2 + epsilon:
        raise ValueError("No solution: M is out of valid range (-2 <= M <= 2).")

    # Handle numerical precision issues
    K = max(min(K, 2), -2)
    M = max(min(M, 2), -2)

    # Compute psi
    psi = math.atan2(K, M)

    # Compute cos(phi)
    denominator = 2 * math.cos(psi)
    if abs(denominator) < epsilon:
        raise ValueError("No solution: Denominator for phi is zero, leading to division by zero.")

    cos_phi = M / denominator

    # Clamp cos_phi to the valid range [-1, 1] to account for numerical errors
    cos_phi = max(min(cos_phi, 1), -1)

    # Compute phi
    phi = math.acos(cos_phi)

    # Compute theta3
    theta3 = 2 * phi

    # Compute theta2
    theta2 = psi - phi

    # Normalize angles to be within [-pi, pi]
    theta1 = math.atan2(math.sin(theta1), math.cos(theta1))
    theta2 = math.atan2(math.sin(theta2), math.cos(theta2))
    theta3 = math.atan2(math.sin(theta3), math.cos(theta3))

    return [theta1, theta2, theta3]


def forward_kinematics(joints: list) -> list:
    """
    Compute the forward kinematics for the 3-DOF robotic chain.

    :param joints: A list of joint angles in radians [theta1, theta2, theta3].
    :return: The position of the end effector in Cartesian coordinates [x, y, z].
    """
    theta1, theta2, theta3 = joints

    # Link lengths
    L1 = 0.4  # Joint1 to Joint2
    L2 = 0.4  # Joint2 to Joint3
    L3 = 0.4  # Joint3 to End Effector

    # Compute sin and cos for efficiency
    sin_theta1 = math.sin(theta1)
    cos_theta1 = math.cos(theta1)
    sin_theta2 = math.sin(theta2)
    cos_theta2 = math.cos(theta2)
    sin_theta3 = math.sin(theta3)
    cos_theta3 = math.cos(theta3)

    # Compute the combined angles
    theta23 = theta2 + theta3
    sin_theta23 = math.sin(theta23)
    cos_theta23 = math.cos(theta23)

    # Compute the projection in the XY-plane
    K = sin_theta2 + sin_theta23
    M = 1 + cos_theta2 + cos_theta23

    # End effector position
    x = L1 * K * sin_theta1
    y = L1 * K * cos_theta1
    z = 0.4 + L1 * cos_theta2 + L2 * cos_theta23 + L3 * cos_theta23

    return [x, y, z]
