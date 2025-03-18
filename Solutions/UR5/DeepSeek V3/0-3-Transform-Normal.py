import numpy as np
import math
from scipy.optimize import minimize

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    target_position = np.array(p)
    target_orientation = np.array(r)
    d2 = 0.13585
    d3 = -0.1197
    a3 = 0.425
    d4 = 0.39225

    def forward_kinematics(theta):
        theta1, theta2, theta3, theta4 = theta
        T01 = np.array([[math.cos(theta1), -math.sin(theta1), 0, 0], [math.sin(theta1), math.cos(theta1), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        T12 = np.array([[math.cos(theta2), 0, math.sin(theta2), 0], [0, 1, 0, d2], [-math.sin(theta2), 0, math.cos(theta2), 0], [0, 0, 0, 1]])
        T23 = np.array([[math.cos(theta3), 0, math.sin(theta3), a3], [0, 1, 0, d3], [-math.sin(theta3), 0, math.cos(theta3), 0], [0, 0, 0, 1]])
        T34 = np.array([[math.cos(theta4), 0, math.sin(theta4), 0], [0, 1, 0, d4], [-math.sin(theta4), 0, math.cos(theta4), 0], [0, 0, 0, 1]])
        T04 = T01 @ T12 @ T23 @ T34
        position = T04[:3, 3]
        orientation = np.array([math.atan2(T04[2, 1], T04[2, 2]), math.asin(-T04[2, 0]), math.atan2(T04[1, 0], T04[0, 0])])
        return (position, orientation)

    def objective(theta):
        position, orientation = forward_kinematics(theta)
        position_error = np.linalg.norm(position - target_position)
        orientation_error = np.linalg.norm(orientation - target_orientation)
        return position_error + orientation_error
    initial_guess = [0.0, 0.0, 0.0, 0.0]
    result = minimize(objective, initial_guess, method='SLSQP')
    return tuple(result.x)