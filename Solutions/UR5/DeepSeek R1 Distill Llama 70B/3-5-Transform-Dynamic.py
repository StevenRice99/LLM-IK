import math
from sympy import symbols, Eq, solve

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Solves for the joint angles θ1, θ2, θ3 to reach the desired position p and orientation r.
    :param p: The desired position [x, y, z].
    :param r: The desired orientation [roll, pitch, yaw] in radians.
    :return: A tuple of joint angles (θ1, θ2, θ3).
    """
    θ1, θ2, θ3 = symbols('θ1 θ2 θ3')
    T1 = [[cos(θ1), -sin(θ1), 0, 0], [sin(θ1), cos(θ1), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    T2 = [[cos(θ2), -sin(θ2), 0, 0], [sin(θ2), cos(θ2), 0, 0], [0, 0, 1, 0.093], [0, 0, 0, 1]]
    T3 = [[cos(θ3), -sin(θ3), 0, 0], [sin(θ3), cos(θ3), 0, 0.093], [0, 0, 1, 0.09465], [0, 0, 0, 1]]
    T_total = multiply_transformations(T1, T2, T3)
    x, y, z = T_total[:3, 3]
    roll, pitch, yaw = extract_orientation(T_total)
    equations = [Eq(x, p[0]), Eq(y, p[1]), Eq(z, p[2]), Eq(roll, r[0]), Eq(pitch, r[1]), Eq(yaw, r[2])]
    solution = solve(equations, (θ1, θ2, θ3))
    return (solution[θ1], solution[θ2], solution[θ3])

def multiply_transformations(*transforms):
    """Multiplies multiple 4x4 transformation matrices."""
    result = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    for transform in transforms:
        result = multiply_matrices(result, transform)
    return result

def multiply_matrices(a, b):
    """Multiplies two 4x4 matrices."""
    result = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                result[i][j] += a[i][k] * b[k][j]
    return result

def extract_orientation(transform):
    """Extracts roll, pitch, yaw from a transformation matrix."""
    roll = math.atan2(transform[1][2], transform[2][2])
    pitch = math.atan2(-transform[0][2], math.sqrt(transform[0][0] ** 2 + transform[0][1] ** 2))
    yaw = math.atan2(transform[1][0], transform[0][0])
    return (roll, pitch, yaw)