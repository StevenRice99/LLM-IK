import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" using the Jacobian inverse method.
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    theta = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    target = np.array([p[0], p[1], p[2]], dtype=float)
    max_iter = 1000
    tolerance = 1e-06
    alpha = 0.5
    for _ in range(max_iter):
        current = forward_kinematics(theta)
        error = target - current
        if np.linalg.norm(error) < tolerance:
            break
        J = jacobian_matrix(theta)
        J_inv = np.linalg.pinv(J)
        theta += alpha * J_inv @ error
    return tuple(theta)

def forward_kinematics(theta: np.ndarray) -> np.ndarray:
    """
    Calculates the end effector position given the joint angles.
    :param theta: The joint angles in radians.
    :return: The position of the end effector as a 3D array [x, y, z].
    """
    link1 = np.array([0, 0, 0, 1])
    link2 = np.array([0, -0.1197, 0.425, 1])
    link3 = np.array([0, 0, 0.39225, 1])
    link4 = np.array([0, 0.093, 0, 1])
    tcp_offset = np.array([0, 0, 0.09465, 1])
    T_cumulative = np.identity(4)
    for i in range(4):
        if i < 3:
            axis = 'Y'
        else:
            axis = 'Z'
        T_joint = transformation_matrix(theta[i], axis)
        T_cumulative = T_cumulative @ T_joint
        if i < 3:
            link = link2 if i == 0 else link3 if i == 1 else link4
            transformed_link = T_joint @ link
        else:
            transformed_link = T_cumulative @ link4
    tcp_transformed = T_cumulative @ tcp_offset
    position = tcp_transformed[:3]
    return position

def transformation_matrix(theta: float, axis: str) -> np.ndarray:
    """
    Creates a 4x4 transformation matrix for a rotation about the specified axis.
    :param theta: The rotation angle in radians.
    :param axis: The axis of rotation ('X', 'Y', 'Z').
    :return: The 4x4 transformation matrix.
    """
    if axis == 'X':
        return np.array([[1, 0, 0, 0], [0, np.cos(theta), -np.sin(theta), 0], [0, np.sin(theta), np.cos(theta), 0], [0, 0, 0, 1]])
    elif axis == 'Y':
        return np.array([[np.cos(theta), 0, np.sin(theta), 0], [0, 1, 0, 0], [-np.sin(theta), 0, np.cos(theta), 0], [0, 0, 0, 1]])
    elif axis == 'Z':
        return np.array([[np.cos(theta), -np.sin(theta), 0, 0], [np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    else:
        raise ValueError('Invalid rotation axis.')

def jacobian_matrix(theta: np.ndarray) -> np.ndarray:
    """
    Computes the Jacobian matrix for the current joint configuration.
    :param theta: The joint angles in radians.
    :return: The 3x4 Jacobian matrix.
    """
    J = np.zeros((3, 4))
    for i in range(4):
        T = forward_kinematics(theta)
        T_i = transformation_matrix(theta[i], 'Y' if i < 3 else 'Z')
        if i < 3:
            axis = np.array([0, 1, 0])
        else:
            axis = np.array([0, 0, 1])
        dT_i = np.zeros((4, 4))
        if i < 3:
            dT_i = np.array([[0, 0, 0, 0], [0, -axis[1], -axis[2], 0], [0, axis[0], axis[1], 0], [0, 0, 0, 1]])
        else:
            dT_i = np.array([[0, -axis[2], axis[1], 0], [axis[2], 0, -axis[0], 0], [-axis[1], axis[0], 0, 0], [0, 0, 0, 1]])
        J[:, i] = (T_i @ dT_i[:, :3, 3]).flatten()
    return J