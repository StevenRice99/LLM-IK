import math
import numpy as np
from scipy.linalg import pinv

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Solves for the joint angles using Jacobian Inverse Kinematics.
    :param p: Target position (x, y, z)
    :param r: Target orientation (x, y, z) in radians
    :return: Joint angles (q1, q2, q3, q4)
    """
    q = np.array([0.0, 0.0, 0.0, 0.0])
    pos_tol = 0.001
    orient_tol = 0.01
    gain = 0.1
    while True:
        current_pose = forward_kinematics(q)
        current_p, current_r = (current_pose[:3], current_pose[3:])
        pos_error = np.array(p) - current_p
        orient_error = np.array(r) - current_r
        if np.all(np.abs(pos_error) < pos_tol) and np.all(np.abs(orient_error) < orient_tol):
            break
        jacobian = compute_jacobian(q)
        j_inv = pinv(jacobian)
        delta_q = gain * np.dot(j_inv, np.concatenate((pos_error, orient_error)))
        q += delta_q
        q = apply_joint_limits(q)
    return tuple(q)

def forward_kinematics(q):
    """
    Computes the TCP pose given joint angles q.
    :param q: Joint angles [q1, q2, q3, q4]
    :return: TCP pose [x, y, z, rx, ry, rz]
    """
    T = np.identity(4)
    for i in range(4):
        axis = get_axis(i)
        T_joint = transformation(q[i], axis, i)
        T = np.dot(T, T_joint)
    x, y, z = T[:3, 3]
    rx, ry, rz = euler_from_rotation_matrix(T[:3, :3])
    return (x, y, z, rx, ry, rz)

def get_axis(joint_index: int) -> str:
    """
    Returns the rotation axis for the specified joint.
    :param joint_index: Index of the joint (0-based)
    :return: Axis as a string ('x', 'y', 'z')
    """
    if joint_index == 0:
        return 'z'
    else:
        return 'y'

def transformation(angle: float, axis: str, joint_index: int) -> np.ndarray:
    """
    Computes the transformation matrix for a joint rotation and translation.
    :param angle: Rotation angle in radians
    :param axis: Axis of rotation ('x', 'y', 'z')
    :param joint_index: Index of the joint (0-based)
    :return: 4x4 transformation matrix
    """
    if axis == 'x':
        rot = np.array([[1, 0, 0, 0], [0, math.cos(angle), -math.sin(angle), 0], [0, math.sin(angle), math.cos(angle), 0], [0, 0, 0, 1]])
    elif axis == 'y':
        rot = np.array([[math.cos(angle), 0, math.sin(angle), 0], [0, 1, 0, 0], [-math.sin(angle), 0, math.cos(angle), 0], [0, 0, 0, 1]])
    elif axis == 'z':
        rot = np.array([[math.cos(angle), -math.sin(angle), 0, 0], [math.sin(angle), math.cos(angle), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    else:
        raise ValueError('Invalid rotation axis')
    if joint_index == 0:
        trans = np.array([0, 0, 0, 1])
    elif joint_index == 1:
        trans = np.array([0, 0.13585, 0, 1])
    elif joint_index == 2:
        trans = np.array([0, -0.1197, 0.425, 1])
    elif joint_index == 3:
        trans = np.array([0, 0, 0.39225, 1])
    else:
        trans = np.array([0, 0, 0, 1])
    T_joint = np.copy(rot)
    T_joint[:3, 3] = trans[:3]
    return T_joint

def euler_from_rotation_matrix(R: np.ndarray) -> tuple[float, float, float]:
    """
    Computes Euler angles (ZYX) from a rotation matrix.
    :param R: 3x3 rotation matrix
    :return: Euler angles (rx, ry, rz) in radians
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-06
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return (x, y, z)

def compute_jacobian(q: np.ndarray) -> np.ndarray:
    """
    Computes the Jacobian matrix for the current joint configuration.
    :param q: Joint angles [q1, q2, q3, q4]
    :return: Jacobian matrix (6x4)
    """
    jacobian = np.zeros((6, 4))
    for i in range(4):
        current_pose = np.array(forward_kinematics(q))
        q_perturbed = q.copy()
        q_perturbed[i] += 1e-06
        perturbed_pose = np.array(forward_kinematics(q_perturbed))
        dx = perturbed_pose[:3] - current_pose[:3]
        dr = perturbed_pose[3:] - current_pose[3:]
        jacobian[:3, i] = dx / 1e-06
        jacobian[3:, i] = dr / 1e-06
    return jacobian

def apply_joint_limits(q: np.ndarray) -> np.ndarray:
    """
    Ensures joint angles stay within their limits.
    :param q: Joint angles [q1, q2, q3, q4]
    :return: Joint angles within limits
    """
    limits = np.array([[-6.2831853, 6.2831853], [-6.2831853, 6.2831853], [-6.2831853, 6.2831853], [-6.2831853, 6.2831853]])
    for i in range(4):
        q[i] = np.clip(q[i], limits[i, 0], limits[i, 1])
    return q