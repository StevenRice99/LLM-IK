import math
from sympy import symbols, Eq, solve, cos, sin

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Solves for the joint angles θ1, θ2, θ3 given the target position p and orientation r.
    :param p: Target position [x, y, z]
    :param r: Target orientation [roll, pitch, yaw] in radians
    :return: Joint angles (θ1, θ2, θ3) in radians
    """
    θ1, θ2, θ3 = symbols('θ1 θ2 θ3')
    T_base_to_joint1 = rotation_matrix_y(θ1)
    T_joint1_to_joint2 = translation_matrix(0, 0, 0.39225) @ rotation_matrix_y(θ2)
    T_joint2_to_joint3 = translation_matrix(0, 0.093, 0) @ rotation_matrix_y(θ2)
    T_joint3_to_tcp = translation_matrix(0, 0, 0.09465) @ rotation_matrix_z(θ3)
    T_total = T_base_to_joint1 @ T_joint1_to_joint2 @ T_joint2_to_joint3 @ T_joint3_to_tcp
    x, y, z = get_position(T_total)
    rx, ry, rz = get_orientation(T_total)
    eq1 = Eq(x, p[0])
    eq2 = Eq(y, p[1])
    eq3 = Eq(z, p[2])
    eq4 = Eq(rx, r[0])
    eq5 = Eq(ry, r[1])
    eq6 = Eq(rz, r[2])
    solution = solve((eq1, eq2, eq3, eq4, eq5, eq6), (θ1, θ2, θ3))
    return (solution[θ1], solution[θ2], solution[θ3])

def rotation_matrix_y(theta):
    return [[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]]

def rotation_matrix_z(theta):
    return [[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]]

def translation_matrix(x, y, z):
    return [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]

def get_position(transformation_matrix):
    return (transformation_matrix[0][3], transformation_matrix[1][3], transformation_matrix[2][3])

def get_orientation(transformation_matrix):
    sy = math.sqrt(transformation_matrix[0][0] ** 2 + transformation_matrix[1][0] ** 2)
    if sy < 1e-06:
        x = math.atan2(transformation_matrix[1][2], transformation_matrix[0][2])
        y = math.atan2(sy, transformation_matrix[2][2])
        z = 0
    else:
        x = math.atan2(transformation_matrix[2][1], transformation_matrix[2][2])
        y = math.atan2(-transformation_matrix[2][0], sy)
        z = math.atan2(transformation_matrix[1][0], transformation_matrix[0][0])
    return (x, y, z)