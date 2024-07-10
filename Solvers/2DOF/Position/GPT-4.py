import numpy as np

def inverse_kinematics(p):
    x, y, z = p
    L1, L2 = 0.4, 0.4  # lengths of the two arm segments

    # Calculate distance from the base to the target position in the XY plane
    d_xy = np.sqrt(x**2 + y**2)
    # Total distance from the base to the target position
    d = np.sqrt(x**2 + y**2 + z**2)

    # Check if the target is reachable
    if d > (L1 + L2):
        raise ValueError("Target position is out of the robot's reach")

    # Calculate theta1 for rotation around z-axis to align with target in XY plane
    theta1 = np.arctan2(y, x)

    # Using cosine rule to find angle at joint 1
    cos_theta2 = (L1**2 + L2**2 - d**2) / (2 * L1 * L2)
    if abs(cos_theta2) > 1:
        raise ValueError("Error in computation: cos(theta2) out of valid range")
    theta2 = np.arccos(cos_theta2)

    # Law of cosines to find the angle at the base joint
    cos_phi = (d**2 + L1**2 - L2**2) / (2 * d * L1)
    if abs(cos_phi) > 1:
        raise ValueError("Error in computation: cos(phi) out of valid range")
    phi = np.arccos(cos_phi)

    # The total angle to lift the arm to the target's z-height
    theta_lift = np.arctan2(z, d_xy)
    theta1_adjust = theta_lift - phi

    return [theta1, theta1_adjust, theta2]
