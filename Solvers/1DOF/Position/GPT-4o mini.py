import math


def inverse_kinematics(p: list) -> list:
    """
    Solve the inverse kinematics to reach the position in the reference frame of the chain's origin.
    :param p: The position for the end effector to reach in a Cartesian [x, y, z].
    :return: A list of the joint values in radians needed for the end effector to reach the position.
    """
    x, y, z = p

    # Fixed distance of 0.4 units in z from Joint 1 to End Effector
    end_effector_z = 0.4

    # Calculate the angle for Joint 1 based on x and y coordinates
    joint_1_angle = math.atan2(y, x)

    # Check if the z-coordinate is within an acceptable range of 0.4
    # Adjust if necessary, as the z-coordinate can vary based on application
    if not (end_effector_z - 0.01 <= z <= end_effector_z + 0.01):
        raise ValueError(f"End effector's z-coordinate should be close to {end_effector_z}.")

    # Return the angle in radians
    return [joint_1_angle]
