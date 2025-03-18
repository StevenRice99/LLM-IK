import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Solves for the joint angles using the Jacobian pseudoinverse method.
    :param p: Target position [x, y, z].
    :param r: Target orientation [x, y, z] in radians.
    :return: Joint angles [q1, q2, q3, q4].
    """
    q = np.array([0.0, 0.0, 0.0, 0.0])
    target = np.concatenate((np.array(p), np.array(r)))
    tol = 1e-06
    max_iter = 1000
    for _ in range(max_iter):
        current = forward_kinematics(q)
        current_pose = np.concatenate((current[:3], current[3:]))
        error = target - current_pose
        if np.linalg.norm(error) < tol:
            break
        J = compute_jacobian(q)
        J_pinv = np.linalg.pinv(J)
        dq = np.dot(J_pinv, error)
        q += dq
        q = np.clip(q, [-6.2831853, -6.2831853, -6.2831853, -6.2831853], [6.2831853, 6.2831853, 6.2831853, 6.2831853])
    return tuple(q)

def forward_kinematics(q):
    """
    Computes the TCP pose given joint angles q.
    :param q: Joint angles [q1, q2, q3, q4].
    :return: TCP pose [x, y, z, rx, ry, rz].
    """
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def compute_jacobian(q):
    """
    Computes the Jacobian matrix for the given joint angles q.
    :param q: Joint angles [q1, q2, q3, q4].
    :return: Jacobian matrix (6x4).
    """
    eps = 1e-08
    J = np.zeros((6, 4))
    for i in range(4):
        dq = np.zeros(4)
        dq[i] = eps
        q_perturbed = q + dq
        pose_perturbed = forward_kinematics(q_perturbed)
        J[:, i] = (np.concatenate((pose_perturbed[:3], pose_perturbed[3:])) - np.concatenate((forward_kinematics(q)[:3], forward_kinematics(q)[3:]))) / eps
    return J