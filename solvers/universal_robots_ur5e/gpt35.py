import numpy as np

def inverse_kinematics(px, py, pz, qw, qx, qy, qz) -> list:
    # DH parameters
    a = [0.0, -0.425, -0.392, 0.0, 0.0, 0.0]
    d = [0.163, 0.0, 0.0, 0.127, 0.1, 0.1]
    alpha = [0, np.pi/2, 0, np.pi/2, 0, 0]

    # Wrist center position
    wc_x = px - d[5] * qw + a[5] * qy
    wc_y = py - d[5] * qx + a[5] * qx
    wc_z = pz - d[5] * qz + a[5] * qz

    # Compute theta1
    theta1 = np.arctan2(wc_y, wc_x)

    # Compute theta3
    D = (wc_x**2 + wc_y**2 + (wc_z - d[0])**2 - a[1]**2 - a[2]**2) / (2 * a[1] * a[2])
    theta3 = np.arctan2(-np.sqrt(1 - D**2), D)

    # Compute theta2
    theta2 = np.arctan2(wc_z - d[0], np.sqrt(wc_x**2 + wc_y**2)) - np.arctan2(a[2] * np.sin(theta3), a[1] + a[2] * np.cos(theta3))

    # Compute orientation angles
    theta4 = np.arctan2(qy, qx)
    theta5 = np.arctan2(-np.sqrt(1 - qz**2), qz)
    theta6 = np.arctan2(qw, -qx)

    return [theta1, theta2, theta3, theta4, theta5, theta6]
