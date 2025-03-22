def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import math
    y_j2 = 0.13585
    y_j3 = -0.1197
    z_j3 = 0.425
    z_j4 = 0.39225
    y_j5 = 0.093
    z_j6 = 0.09465
    y_tcp = 0.0823
    x_target, y_target, z_target = p
    q1 = math.atan2(x_target, y_target)
    c1 = math.cos(q1)
    s1 = math.sin(q1)
    x_local = c1 * x_target + s1 * y_target
    y_local = -s1 * x_target + c1 * y_target
    z_local = z_target
    wrist_y = y_local - y_tcp
    wrist_x = x_local
    wrist_z = z_local
    wrist_y -= y_j2 + y_j3 + y_j5
    wrist_z -= z_j3 + z_j4 + z_j6
    d = math.sqrt(wrist_y ** 2 + wrist_z ** 2)
    a1 = math.sqrt(z_j3 ** 2 + y_j3 ** 2)
    a2 = math.sqrt(z_j4 ** 2 + y_j5 ** 2)
    phi = math.atan2(wrist_z, wrist_y)
    cos_q3 = (d ** 2 - a1 ** 2 - a2 ** 2) / (2 * a1 * a2)
    if cos_q3 < -1 or cos_q3 > 1:
        q2 = -math.pi / 4
        q3 = math.pi / 2
        q4 = 0
        q5 = 0
        q6 = 0
        return (q1, q2, q3, q4, q5, q6)
    q3 = math.acos(cos_q3)
    beta = math.atan2(a2 * math.sin(q3), a1 + a2 * math.cos(q3))
    q2 = phi - beta
    q4 = -q2 - q3
    q5 = 0
    q6 = 0

    def normalize(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    q1 = normalize(q1)
    q2 = normalize(q2)
    q3 = normalize(q3)
    q4 = normalize(q4)
    q5 = normalize(q5)
    q6 = normalize(q6)
    return (q1, q2, q3, q4, q5, q6)