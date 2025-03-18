import sympy as sp

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    theta1, theta2, theta3, theta4 = sp.symbols('theta1 theta2 theta3 theta4')
    T1 = sp.Matrix([[sp.cos(theta1), -sp.sin(theta1), 0, 0], [sp.sin(theta1), sp.cos(theta1), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T2 = sp.Matrix([[sp.cos(theta2 + sp.pi / 2), 0, sp.sin(theta2 + sp.pi / 2), 0.13585 * sp.cos(theta2 + sp.pi / 2)], [sp.sin(theta2 + sp.pi / 2), 0, -sp.cos(theta2 + sp.pi / 2), 0.13585 * sp.sin(theta2 + sp.pi / 2)], [0, 1, 0, 0.425], [0, 0, 0, 1]])
    T3 = sp.Matrix([[sp.cos(theta3 + sp.pi / 2), 0, sp.sin(theta3 + sp.pi / 2), -0.1197 * sp.cos(theta3 + sp.pi / 2)], [sp.sin(theta3 + sp.pi / 2), 0, -sp.cos(theta3 + sp.pi / 2), -0.1197 * sp.sin(theta3 + sp.pi / 2)], [0, 1, 0, 0.39225], [0, 0, 0, 1]])
    T4 = sp.Matrix([[sp.cos(theta4 + sp.pi / 2), 0, sp.sin(theta4 + sp.pi / 2), 0], [sp.sin(theta4 + sp.pi / 2), 0, -sp.cos(theta4 + sp.pi / 2), 0], [0, 1, 0, 0.81725], [0, 0, 0, 1]])
    T_total = T1 * T2 * T3 * T4
    x_tcp = T_total[0, 3]
    y_tcp = T_total[1, 3]
    z_tcp = T_total[2, 3]
    eq1 = sp.Eq(x_tcp, p[0])
    eq2 = sp.Eq(y_tcp, p[1])
    eq3 = sp.Eq(z_tcp, p[2])
    solution = sp.solve((eq1, eq2, eq3), (theta1, theta2, theta3, theta4))
    return (float(solution[theta1]), float(solution[theta2]), float(solution[theta3]), float(solution[theta4]))