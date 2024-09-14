import math

def inverse_kinematics(p):
    x, y, z = p
    L1 = 0.4  # Base height (from base to Joint 2)
    L2 = 0.4  # Length from Joint 2 to Joint 3
    L3 = 0.4  # Length from Joint 3 to End Effector
    L4 = 0.4  # Length from Joint 3 to End Effector along z

    # Step 1: Compute theta1
    theta1 = math.atan2(y, x)

    # Step 2: Compute r and h
    r = math.hypot(x, y)
    h = z - L1 - L4  # Adjust for base height and end effector offset

    # Step 3: Compute D
    num = r**2 + h**2 - L2**2 - L3**2
    den = 2 * L2 * L3
    D = num / den

    # Check reachability
    if abs(D) > 1.0:
        raise ValueError("Position not reachable")

    # Step 4: Compute theta3 (two solutions)
    theta3_a = math.acos(D)
    theta3_b = -math.acos(D)

    # Step 5: Compute theta2 for each theta3
    def compute_theta2(theta3):
        k1 = L2 + L3 * math.cos(theta3)
        k2 = L3 * math.sin(theta3)
        phi = math.atan2(h, r)
        psi = math.atan2(k2, k1)
        theta2 = phi - psi
        return theta2

    theta2_a = compute_theta2(theta3_a)
    theta2_b = compute_theta2(theta3_b)

    # Choose one solution (e.g., elbow-down configuration)
    theta2 = theta2_a
    theta3 = theta3_a

    # Return the joint angles
    return [theta1, theta2, theta3]
