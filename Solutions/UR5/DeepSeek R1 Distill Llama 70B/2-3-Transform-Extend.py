import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [roll, pitch, yaw].
    :return: A tuple of the values to set the joints to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    rx, ry, rz = r
    q1 = np.arctan2(x, z)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    theta_y = np.arctan2(R[0, 2], R[0, 0])
    q2 = theta_y - q1
    return (q1, q2)