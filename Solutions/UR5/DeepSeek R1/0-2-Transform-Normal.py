import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    rx, ry, rz = r
    x, y, z = p
    theta1 = np.arctan2(y, x)
    dx = x - 0.13585 * np.sin(theta1)
    dy = y - 0.13585 * np.cos(theta1)
    dz = z
    A = np.hypot(dx, dy)
    B = dz
    a2 = 0.425
    a3 = 0.39225
    cos_theta3 = (A ** 2 + B ** 2 - a2 ** 2 - a3 ** 2) / (2 * a2 * a3)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = np.arccos(cos_theta3)
    theta3 = np.array([theta3, -theta3])
    alpha = np.arctan2(B, A)
    theta2 = alpha - np.arctan2(a3 * np.sin(theta3), a2 + a3 * np.cos(theta3))
    theta23 = theta2 + theta3
    error = np.abs(theta23 - ry)
    idx = np.argmin(error)
    theta2_val = theta2[idx]
    theta3_val = ry - theta2_val
    return (theta1, theta2_val, theta3_val)