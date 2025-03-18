import numpy as np
from scipy.spatial.transform import Rotation as R

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x, y, z = p
    roll, pitch, yaw = r
    rot_matrix = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    d2 = -0.1197
    d3 = 0.39225
    d4 = 0.093
    d5 = 0.09465
    q1 = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)
    h = z - d2
    q2_plus_q3 = np.arctan2(r, h)
    q3 = np.arccos((r ** 2 + h ** 2 - (d3 + d4 + d5) ** 2) / (2 * (d3 + d4 + d5) * np.sqrt(r ** 2 + h ** 2)))
    q2 = q2_plus_q3 - q3
    R1 = R.from_euler('y', q1).as_matrix()
    R2 = R.from_euler('y', q2).as_matrix()
    R3 = R.from_euler('y', q3).as_matrix()
    R_up_to_3 = R1 @ R2 @ R3
    R4 = R_up_to_3.T @ rot_matrix
    q4 = np.arctan2(R4[1, 0], R4[0, 0])
    return (q1, q2, q3, q4)