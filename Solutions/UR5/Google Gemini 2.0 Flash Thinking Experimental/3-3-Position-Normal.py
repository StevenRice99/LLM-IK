def inverse_kinematics(p: tuple[float, float, float]) -> float:
    """
    Gets the joint values needed to reach position "p".
    :param p :The position to reach in the form [x, y, z].
    :return: The value to set the link to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    joint_value = np.arctan2(-z, x)
    return float(joint_value)