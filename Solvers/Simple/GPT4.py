def inverse_kinematics(p: list, q: list or None = None) -> list:
    """
    Solve the inverse kinematics to reach position and orientation, both in the reference frame of the chain's origin.
    :param p: The position for the end effector to reach in a Cartesian [x, y, z].
    :param q: The orientation for the end effector to reach in radians in a quaternion [w, x, y, z].
    If "None" is passed for the orientation, then only solve for inverse kinematics for the position.
    :return: A list of the joint values in radians needed for the end effector to reach the position and orientation.
    """
    return []
