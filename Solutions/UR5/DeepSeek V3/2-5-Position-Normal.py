import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x, y, z = p
    theta1 = np.arctan2(x, z)
    d = np.sqrt(y ** 2 + (z * np.cos(theta1) - x * np.sin(theta1)) ** 2)
    theta2 = np.arctan2(y, z * np.cos(theta1) - x * np.sin(theta1))
    theta3 = np.arctan2(y, x)
    theta4 = np.arctan2(z * np.cos(theta1) - x * np.sin(theta1), y)
    return (theta1, theta2, theta3, theta4)