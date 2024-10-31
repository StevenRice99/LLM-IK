import math


def inverse_kinematics(p: list) -> list[float]:
    """
    Solve the inverse kinematics to reach the position and orientation, both in the reference frame of the chain's origin.
    :param p: The position for the end effector to reach in a Cartesian [x, y, z].
    :return: A list of the joint values in radians needed for the end effector to reach the position.
    """
    _, y, z = p

    # Calculate theta based on the given y and z position
    theta1 = math.atan2(y, z)  # One possible solution
    theta2 = theta1 + math.pi if theta1 < 0 else theta1 - math.pi  # The symmetric solution

    # Calculate the resulting positions for both theta1 and theta2
    y1, z1 = 0.4 * math.sin(theta1), 0.4 * math.cos(theta1)
    y2, z2 = 0.4 * math.sin(theta2), 0.4 * math.cos(theta2)

    # Check which solution matches the target position within tolerance
    if math.isclose(y, y1, rel_tol=1e-5) and math.isclose(z, z1, rel_tol=1e-5):
        return [theta1]
    elif math.isclose(y, y2, rel_tol=1e-5) and math.isclose(z, z2, rel_tol=1e-5):
        return [theta2]
    else:
        raise ValueError("No valid solution found within tolerance.")
