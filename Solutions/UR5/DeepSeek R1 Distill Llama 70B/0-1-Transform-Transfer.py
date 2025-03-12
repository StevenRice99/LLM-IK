import math
import numpy as np
import sympy as sp

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Computes the joint angles theta1 and theta2 to reach the given position p and orientation r.
    
    Args:
        p: A tuple (x, y, z) representing the target position.
        r: A tuple (x, y, z) representing the target orientation in Euler angles (radians).
    
    Returns:
        A tuple (theta1, theta2) in radians.
    """
    theta1, theta2 = sp.symbols('theta1 theta2')
    Rz = sp.Matrix([[sp.cos(theta1), -sp.sin(theta1), 0], [sp.sin(theta1), sp.cos(theta1), 0], [0, 0, 1]])
    Ry = sp.Matrix([[sp.cos(theta2), 0, sp.sin(theta2)], [0, 1, 0], [-sp.sin(theta2), 0, sp.cos(theta2)]])
    R_total = Rz * Ry
    roll, pitch, yaw = r
    R_target = sp.Matrix([[sp.cos(yaw) * sp.cos(pitch), sp.sin(yaw) * sp.cos(pitch), -sp.sin(pitch)], [sp.sin(yaw) * sp.cos(pitch), sp.cos(yaw) * sp.cos(pitch), sp.sin(yaw) * sp.sin(pitch)], [sp.sin(yaw) * sp.sin(pitch), sp.cos(yaw) * sp.sin(pitch), sp.cos(pitch)]])
    equations = []
    for i in range(3):
        for j in range(3):
            equations.append(sp.Eq(R_total[i, j], R_target[i, j]))
    solution = sp.solve(equations, (theta1, theta2))
    if solution:
        theta1_val = float(solution[0][0])
        theta2_val = float(solution[0][1])
    else:
        theta1_val = 0.0
        theta2_val = 0.0
    return (theta1_val, theta2_val)