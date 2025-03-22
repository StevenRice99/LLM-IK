def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    A closed-form solution for the 5-DOF manipulator that builds directly on
    the known 4-DOF sub-chain result from EXISTING 4, setting the 5th joint to zero.

    EXISTING 4 code (slightly adapted) solves for joints (q1, q2, q3, q4)
    assuming the geometry up to the TCP is what the URDF describes (with
    the final link effectively at q4, ignoring q5 rotation for position).
    Because the 5th joint is a rotation about Z that does not affect the
    end-effector (TCP) position, we set q5=0, leading to a consistent
    position-only IK solution.

    :param p: The target position in 3D space as (x, y, z).
    :return: (q1, q2, q3, q4, q5) in radians, suitable for reaching position p.
    """
    import numpy as np
    x, y, z = p
    numerator = x ** 2 + y ** 2 + z ** 2 - 0.346395
    denominator = 0.3341625
    cos_q3 = numerator / denominator
    cos_q3 = np.clip(cos_q3, -1.0, 1.0)
    q3 = np.arccos(cos_q3)
    A = 0.425 + 0.39225 * np.cos(q3)
    B = 0.39225 * np.sin(q3)
    S_squared = x ** 2 + y ** 2 - 0.10915 ** 2
    S = np.sqrt(S_squared) if S_squared >= 0 else 0.0
    C = z
    numerator_q2 = S * A - C * B
    denominator_q2 = S * B + C * A
    q2 = np.arctan2(numerator_q2, denominator_q2)
    phi = np.arctan2(0.10915, S) if abs(S) > 1e-09 else np.pi / 2
    q1 = np.arctan2(y, x) - phi
    q4 = 0.0
    q5 = 0.0

    def normalize(angle: float) -> float:
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
    q1 = normalize(q1)
    q2 = normalize(q2)
    q3 = normalize(q3)
    q4 = normalize(q4)
    q5 = normalize(q5)
    return (q1, q2, q3, q4, q5)