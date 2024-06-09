import numpy as np

def inverse_kinematics(p):
    # Desired position in local coordinates of Joint 1
    p_local = [p[0], p[1], p[2] - 0.4]  # Adjust for the offset of the end effector

    # Calculate the joint angle
    theta = np.arctan2(p_local[1], p_local[2])

    # Check if the calculated angle puts the end effector in the correct position
    # If not, adjust the angle by pi
    if not np.allclose(forward_kinematics(theta), p):
        theta += np.pi

    # Return joint values
    return [theta]

def forward_kinematics(theta):
    # Calculate the position of the end effector given the joint angle
    return [0, np.sin(theta), 0.4 + np.cos(theta)]