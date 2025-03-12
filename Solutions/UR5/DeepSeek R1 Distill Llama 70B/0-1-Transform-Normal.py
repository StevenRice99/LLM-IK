import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [roll, pitch, yaw].
    :return: A tuple of the values to set the joints to for reaching position "p" and orientation "r".
    """
    p_x, p_y, p_z = p
    roll, pitch, yaw = r
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    R_desired = np.dot(Rz, np.dot(Ry, Rx))
    g = R_desired[2, 0]
    i = R_desired[2, 2]
    theta2 = np.arctan2(-g, i)
    cos_theta1 = R_desired[0, 0] / np.cos(theta2)
    sin_theta1 = R_desired[0, 2] / np.cos(theta2)
    theta1 = np.arctan2(sin_theta1, cos_theta1)
    return (theta1, theta2)