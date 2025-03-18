import numpy as np
from scipy.linalg import inv
from sympy import symbols, Eq, solve

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    dh_parameters = [{'a': 0, 'alpha': np.pi / 2, 'd': 0}, {'a': 0.425, 'alpha': 0, 'd': -0.1197}, {'a': 0.39225, 'alpha': 0, 'd': 0}, {'a': 0, 'alpha': np.pi / 2, 'd': 0.093}, {'a': 0.09465, 'alpha': 0, 'd': 0}]
    q1, q2, q3, q4, q5 = symbols('q1 q2 q3 q4 q5')
    T1 = construct_transformation_matrix(q1, dh_parameters[0])
    T2 = construct_transformation_matrix(q2, dh_parameters[1])
    T3 = construct_transformation_matrix(q3, dh_parameters[2])
    T4 = construct_transformation_matrix(q4, dh_parameters[3])
    T5 = construct_transformation_matrix(q5, dh_parameters[4])
    T_total = np.dot(np.dot(np.dot(np.dot(T1, T2), T3), T4), T5)
    T_desired = construct_desired_transformation(p, r)
    equations = equate_transformations(T_total, T_desired)
    solution = solve(equations, (q1, q2, q3, q4, q5))
    return tuple(solution.values())

def construct_transformation_matrix(theta, dh_params):
    a = dh_params['a']
    alpha = dh_params['alpha']
    d = dh_params['d']
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    T = np.array([[cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta], [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta], [0, sin_alpha, cos_alpha, d], [0, 0, 0, 1]])
    return T

def construct_desired_transformation(p, r):
    r_matrix = rotation_matrix_from_rpy(r)
    T = np.eye(4)
    T[:3, :3] = r_matrix
    T[:3, 3] = p
    return T

def rotation_matrix_from_rpy(rpy):
    roll, pitch, yaw = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return np.dot(np.dot(Rx, Ry), Rz)

def equate_transformations(T_total, T_desired):
    equations = []
    for i in range(3):
        for j in range(3):
            if i == j:
                equations.append(Eq(T_total[i, j], T_desired[i, j]))
            else:
                equations.append(Eq(T_total[i, j], T_desired[i, j]))
    return equations