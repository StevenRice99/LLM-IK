import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Computes the joint angles needed to reach the desired position and orientation.
    
    Args:
        p: The desired position as a tuple (x, y, z).
        r: The desired orientation as a tuple of Euler angles (r_x, r_y, r_z) in radians.
    
    Returns:
        A tuple of joint angles (θ1, θ2) in radians.
    """
    r_x, r_y, r_z = r
    R_x = np.array([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)], [0, np.sin(r_x), np.cos(r_x)]])
    R_y = np.array([[np.cos(r_y), 0, np.sin(r_y)], [0, 1, 0], [-np.sin(r_y), 0, np.cos(r_y)]])
    R_z = np.array([[np.cos(r_z), -np.sin(r_z), 0], [np.sin(r_z), np.cos(r_z), 0], [0, 0, 1]])
    R_desired = np.dot(R_z, np.dot(R_y, R_x))
    sinθ2 = -R_desired[0, 1]
    cosθ2 = R_desired[1, 1]
    θ2 = np.arctan2(sinθ2, cosθ2)
    cosθ1 = R_desired[0, 0] / cosθ2
    sinθ1 = R_desired[0, 2] / cosθ2
    θ1 = np.arctan2(sinθ1, cosθ1)
    θ1 = np.clip(θ1, -6.2831853, 6.2831853)
    θ2 = np.clip(θ2, -6.2831853, 6.2831853)
    return (float(θ1), float(θ2))