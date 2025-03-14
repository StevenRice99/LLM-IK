import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r" using Jacobian inverse.
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    L1 = 0.13585
    L2 = 0.425
    TCP_OFFSET = np.array([0, -0.1197, 0.425])
    theta1 = 0.0
    theta2 = 0.0

    def jacobian(theta1, theta2):
        Jp = np.zeros((3, 2))
        Jp[:, 0] = np.array([-L1 * np.sin(theta1) - L2 * np.sin(theta1 + theta2), L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2), 0])
        Jp[:, 1] = np.array([-L2 * np.sin(theta1 + theta2), L2 * np.cos(theta1 + theta2), 0])
        Jr = np.zeros((3, 2))
        Jr[:, 0] = np.array([1, 0, 0])
        Jr[:, 1] = np.array([0, 0, 0])
        J = np.vstack((Jp, Jr))
        return J
    target_position = np.array(p)
    target_orientation = np.array(r)

    def forward_kinematics(theta1, theta2):
        x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
        y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
        z = 0
        position = np.array([x, y, z]) + TCP_OFFSET
        orientation = np.array([theta1, 0, 0])
        return (position, orientation)
    current_position, current_orientation = forward_kinematics(theta1, theta2)
    error_position = target_position - current_position
    error_orientation = target_orientation - current_orientation
    J = jacobian(theta1, theta2)
    J_inv = np.linalg.pinv(J)
    joint_rates = J_inv @ np.concatenate((error_position, error_orientation))
    theta1 += joint_rates[0]
    theta2 += joint_rates[1]
    return (theta1, theta2)