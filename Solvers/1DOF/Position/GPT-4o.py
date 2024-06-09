import numpy as np


def inverse_kinematics(p: list) -> list:
    """
    Solve the inverse kinematics to reach the position in the reference frame of the chain's origin.
    :param p: The position for the end effector to reach in a Cartesian [x, y, z].
    :return: A list of the joint values in radians needed for the end effector to reach the position.
    """
    # Desired position for the end effector
    x, y, z = p

    # The length of the arm (distance from Joint 1 to End Effector)
    l = 0.4

    # Calculate the distance in the yz-plane from the origin to the desired position
    d = np.sqrt(y ** 2 + z ** 2)

    # Ensure the position is reachable
    if d > l:
        raise ValueError("The target position is out of reach.")

    # Calculate the rotation angle around the x-axis (Joint 1)
    theta1 = np.arctan2(z, y)

    return [theta1]