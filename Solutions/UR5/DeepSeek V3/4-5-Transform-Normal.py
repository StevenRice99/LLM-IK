import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    x, y, z = p
    r_x, r_y, r_z = r
    z_offset = z - 0.09465
    theta2 = np.arccos(z_offset / 0.0823)
    theta1 = np.arctan2(y, x)
    theta2 = r_y
    return (theta1, theta2)