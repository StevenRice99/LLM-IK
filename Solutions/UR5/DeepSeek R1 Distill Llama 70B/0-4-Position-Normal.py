from sympy import symbols, sin, cos, Matrix, Eq, solve

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    θ1, θ2, θ3, θ4, θ5 = symbols('θ1 θ2 θ3 θ4 θ5')
    T1 = Matrix([[cos(θ1), -sin(θ1), 0, 0], [sin(θ1), cos(θ1), 0, 0.13585], [0, 0, 1, 0], [0, 0, 0, 1]])
    T2 = Matrix([[cos(θ2), 0, sin(θ2), 0], [0, 1, 0, 0], [-sin(θ2), 0, cos(θ2), 0.425], [0, 0, 0, 1]])
    T3 = Matrix([[cos(θ3), 0, sin(θ3), 0], [0, 1, 0, 0], [-sin(θ3), 0, cos(θ3), 0.39225], [0, 0, 0, 1]])
    T4 = Matrix([[cos(θ4), 0, sin(θ4), 0], [0, 1, 0, 0], [-sin(θ4), 0, cos(θ4), 0], [0, 0, 0, 1]])
    T5 = Matrix([[cos(θ5), -sin(θ5), 0, 0], [sin(θ5), cos(θ5), 0, 0.093], [0, 0, 1, 0.09465], [0, 0, 0, 1]])
    T_total = T1 * T2 * T3 * T4 * T5
    x, y, z = (T_total[0, 3], T_total[1, 3], T_total[2, 3])
    eq1 = Eq(x, p[0])
    eq2 = Eq(y, p[1])
    eq3 = Eq(z, p[2])
    solution = solve((eq1, eq2, eq3), (θ1, θ2, θ3, θ4, θ5))
    return (float(solution[θ1]), float(solution[θ2]), float(solution[θ3]), float(solution[θ4]), float(solution[θ5]))