import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Computes the two revolute angles (q1 about Z, q2 about Y) that
    drive the chain:
      R = Rz(q1) · Ry(q2)
    to the desired TCP orientation R_des given in URDF‐style RPY = r,
    and (since the pose is assumed reachable) simultaneously yields
    the correct TCP position.

    :param p:  target position [x, y, z] (unused in closed‐form; assumed
               consistent with the orientation)
    :param r:  target orientation [roll, pitch, yaw] in radians
    :returns: (q1, q2) in radians, each wrapped to (–π, +π]
    """
    roll, pitch, yaw = r
    roll = (roll + math.pi) % (2 * math.pi) - math.pi
    pitch = (pitch + math.pi) % (2 * math.pi) - math.pi
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    tol = 1e-06
    if abs(roll) < tol:
        q1 = yaw
        q2 = pitch
    elif abs(abs(roll) - math.pi) < tol:
        s = math.copysign(math.pi, roll)
        q1 = yaw - s
        q2 = s - pitch
    else:
        q1 = yaw
        q2 = pitch
    q1 = (q1 + math.pi) % (2 * math.pi) - math.pi
    q2 = (q2 + math.pi) % (2 * math.pi) - math.pi
    return (q1, q2)