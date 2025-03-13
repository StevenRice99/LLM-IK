import numpy as np
from sympy import symbols, Eq, solve

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Solves for the joint angles needed to reach the position p using inverse kinematics.
    :param p: The desired position [x, y, z].
    :return: A tuple of joint angles (θ1, θ2, θ3) in radians.
    """
    x_des, y_des, z_des = p
    theta1, theta2, theta3 = symbols('theta1 theta2 theta3')
    eq1 = Eq(x_expr, x_des)
    eq2 = Eq(y_expr, y_des)
    eq3 = Eq(z_expr, z_des)
    solution = solve((eq1, eq2, eq3), (theta1, theta2, theta3))
    theta1_val = solution[theta1]
    theta2_val = solution[theta2]
    theta3_val = solution[theta3]
    return (theta1_val, theta2_val, theta3_val)