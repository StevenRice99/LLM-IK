def inverse_kinematics(p):
    x, y, z = p
    import math
    L1 = 0.4  # Length from Base to Joint 2
    L2 = 0.4  # Length from Joint 2 to End Effector

    # Check if the position is within reach
    if (x**2 + y**2 + (z - L1)**2) > (L2 + 1e-6)**2:
        raise ValueError("Position is out of reach")

    # Compute q1
    q1 = math.atan2(-x, y)

    # Compute cos(q2)
    cos_q2 = (z - L1) / L2
    if abs(cos_q2) > 1:
        raise ValueError("No solution exists for the given position")

    # Compute both possible sin(q2)
    sin_q2_positive = math.sqrt(1 - cos_q2**2)
    sin_q2_negative = -sin_q2_positive

    # Possible q2 angles
    q2_options = [
        math.atan2(sin_q2_positive, cos_q2),
        math.atan2(sin_q2_negative, cos_q2)
    ]

    # Function to compute the forward kinematics
    def forward_kinematics(q1, q2):
        x_fk = L2 * math.sin(q1) * math.sin(q2)
        y_fk = -L2 * math.cos(q1) * math.sin(q2)
        z_fk = L2 * cos_q2 + L1
        return x_fk, y_fk, z_fk

    # Choose the q2 that results in the correct position
    min_error = float('inf')
    best_q2 = None
    for q2 in q2_options:
        x_fk, y_fk, z_fk = forward_kinematics(q1, q2)
        error = (x_fk - x)**2 + (y_fk - y)**2 + (z_fk - z)**2
        if error < min_error:
            min_error = error
            best_q2 = q2

    q2 = best_q2
    return [q1, q2]
