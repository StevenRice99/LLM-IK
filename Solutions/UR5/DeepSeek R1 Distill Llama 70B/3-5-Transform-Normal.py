def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Solves the inverse kinematics problem for a 3-DOF serial manipulator.
    
    Args:
        p: The desired position of the TCP in the form [x, y, z].
        r: The desired orientation of the TCP as Euler angles [roll, pitch, yaw] in radians.
        
    Returns:
        A tuple of three joint angles [q1, q2, q3] in radians.
    """
    q1, q2, q3 = sp.symbols('q1 q2 q3')
    T1 = sp.Matrix([[sp.cos(q1), -sp.sin(q1), 0, 0], [sp.sin(q1), sp.cos(q1), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T2 = sp.Matrix([[sp.cos(q2), -sp.sin(q2), 0, 0], [sp.sin(q2), sp.cos(q2), 0, 0.093], [0, 0, 1, 0], [0, 0, 0, 1]])
    T3 = sp.Matrix([[sp.cos(q3), -sp.sin(q3), 0, 0], [sp.sin(q3), sp.cos(q3), 0, 0], [0, 0, 1, 0.09465], [0, 0, 0, 1]])
    T_tcp = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0.0823], [0, 0, 1, 0], [0, 0, 0, 1]])
    T_total = T1 * T2 * T3 * T_tcp
    r_matrix = Rotation.from_euler('xyz', r, degrees=False).as_matrix()
    T_desired = sp.Matrix([[r_matrix[0, 0], r_matrix[0, 1], r_matrix[0, 2], p[0]], [r_matrix[1, 0], r_matrix[1, 1], r_matrix[1, 2], p[1]], [r_matrix[2, 0], r_matrix[2, 1], r_matrix[2, 2], p[2]], [0, 0, 0, 1]])
    equations = []
    for i in range(3):
        for j in range(4):
            if j < 3:
                equations.append(sp.Eq(T_total[i, j], T_desired[i, j]))
            else:
                pass
    solution = sp.solve(equations, (q1, q2, q3))
    if isinstance(solution, list):
        solution = solution[0]
    q1_val = solution[q1]
    q2_val = solution[q2]
    q3_val = solution[q3]
    return (float(q1_val), float(q2_val), float(q3_val))