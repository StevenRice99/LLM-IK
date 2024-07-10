import numpy as np


def inverse_kinematics(p: list) -> list:
    """
    Solve the inverse kinematics to reach the position in the reference frame of the chain's origin.
    :param p: The position for the end effector to reach in a Cartesian [x, y, z].
    :return: A list of the joint values in radians needed for the end effector to reach the position.
    """
    x, y, z = p

    # The end effector position in the local frame
    z_offset = 0.4

    if abs(z_offset) < 1e-6:
        raise ValueError("The z offset cannot be zero.")

    # Calculate the potential angles
    theta_1 = np.arctan2(y, z)
    theta_2 = np.arctan2(-y, z)

    # Verify which angle satisfies the desired position
    possible_solutions = [
        theta_1 if np.isclose(np.sin(theta_1) * z_offset, y) and np.isclose(np.cos(theta_1) * z_offset, z) else None,
        theta_2 if np.isclose(np.sin(theta_2) * z_offset, y) and np.isclose(np.cos(theta_2) * z_offset, z) else None,
    ]

    # Filter out invalid solutions
    valid_solutions = [theta for theta in possible_solutions if theta is not None]

    if not valid_solutions:
        raise ValueError("No valid solution found.")

    return valid_solutions
