import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Inverse kinematics for 5‑DOF:
      J1: Rot Y at [0,0,0]
      J2: Rot Y at [0,-0.1197,0.425]
      J3: Rot Y at [0,0,0.39225]
      J4: Rot Z at [0,0.093,0]
      J5: Rot Y at [0,0,0.09465]
      TCP: Trans [0,0.0823,0], Rot Z +90°
    Returns (θ1,θ2,θ3,θ4,θ5).
    """
    x_t, y_t, z_t = p
    roll, pitch, yaw = r
    cr, sr = (math.cos(roll), math.sin(roll))
    cp, sp = (math.cos(pitch), math.sin(pitch))
    cy, sy = (math.cos(yaw), math.sin(yaw))
    R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R_tgt = R_z @ R_y @ R_x
    p_tcp = np.array([0.0, 0.0823, 0.0])
    R_tcp = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    p05 = np.array([x_t, y_t, z_t]) - R_tgt @ p_tcp
    R05 = R_tgt @ R_tcp.T
    L5 = 0.09465
    p04 = p05 - R05[:, 2] * L5
    r = R05
    phi = math.atan2(r[2, 1], -r[0, 1])
    theta4 = math.acos(max(-1.0, min(1.0, r[1, 1])))
    theta5 = math.atan2(r[1, 2], r[1, 0])
    p03 = p04 - np.array([0.0, 0.093, 0.0])
    x3, _, z3 = p03
    a = 0.425
    b = 0.39225
    D = (x3 * x3 + z3 * z3 - a * a - b * b) / (2 * a * b)
    D = max(-1.0, min(1.0, D))
    sol12 = []
    for sgn in [+1.0, -1.0]:
        th2 = sgn * math.acos(D)
        num = b * math.sin(th2)
        den = a + b * math.cos(th2)
        th1 = math.atan2(x3, z3) - math.atan2(num, den)
        x_c = a * math.sin(th1) + b * math.sin(th1 + th2)
        z_c = a * math.cos(th1) + b * math.cos(th1 + th2)
        if math.isclose(x_c, x3, abs_tol=1e-06) and math.isclose(z_c, z3, abs_tol=1e-06):
            sol12.append((th1, th2))
    if sol12:
        theta1, theta2 = sol12[0]
    else:
        theta1, theta2 = (0.0, 0.0)
    theta3 = phi - theta1 - theta2
    return (theta1, theta2, theta3, theta4, theta5)