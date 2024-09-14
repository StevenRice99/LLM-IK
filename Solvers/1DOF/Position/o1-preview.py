import math

def inverse_kinematics(p: list) -> list:
    p_x, p_y, p_z = p

    # The robot cannot reach positions with a non-zero x-coordinate
    if abs(p_x) > 1e-6:
        raise ValueError("No solution: x component must be zero.")

    # Length of the link
    reach = 0.4

    # Check if the point is within reach (on the circle of radius 0.4 in the yz-plane)
    distance_squared = p_y**2 + p_z**2
    if abs(distance_squared - reach**2) > 1e-6:
        raise ValueError("No solution: point is not reachable.")

    # Calculate sin(theta) and cos(theta)
    sin_theta = -p_y / reach
    cos_theta = p_z / reach

    # Normalize to handle numerical errors
    magnitude = math.hypot(sin_theta, cos_theta)
    sin_theta /= magnitude
    cos_theta /= magnitude

    # Calculate the joint angle theta
    theta = math.atan2(sin_theta, cos_theta)

    return [theta]
