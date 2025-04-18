import math
from typing import Tuple

def inverse_kinematics(p: Tuple[float, float, float], r: Tuple[float, float, float]) -> Tuple[float, float]:
    """
    Analytic IK for the 2‐DOF arm:
      Joint1: Revolute about Z at base
      Link1:  translate (0, 0.13585, 0)
      Joint2: Revolute about Y
      Link2:  translate (0, –0.1197, 0.425) → TCP

    :param p: Desired TCP position (x,y,z)
    :param r: Desired TCP orientation RPY (roll, pitch, yaw)
    :return: (q1, q2) in radians
    """
    x, y, z = p
    roll, pitch, yaw = r
    eps = 1e-08
    if abs(roll) < eps:
        q2 = pitch
        q1 = yaw
    elif abs(abs(roll) - math.pi) < eps:
        q2 = math.pi - pitch
        if roll > 0:
            q1 = yaw - math.pi
        else:
            q1 = yaw + math.pi
    else:
        q2 = pitch
        q1 = yaw
    q1 = math.atan2(math.sin(q1), math.cos(q1))
    return (q1, q2)