import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    L1 = 0.093
    L2 = 0.09465
    theta2_1 = np.arcsin(z / L2)
    theta2_2 = np.pi - theta2_1

    def solve_theta1(theta2):
        denominator = L1 + L2 * np.cos(theta2)
        sin_theta1 = y / denominator
        cos_theta1 = x / (L2 * np.cos(theta2))
        return np.arctan2(sin_theta1, cos_theta1)
    theta1_1 = solve_theta1(theta2_1)
    theta1_2 = solve_theta1(theta2_2)
    return (theta1_1, theta2_1)