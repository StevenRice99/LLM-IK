import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Inverse kinematics for 3‑DOF:
      Joint1: revolute about Z at [0,0,0]
      Joint2: revolute about Y at [0,0.13585,0]
      Joint3: revolute about Y at [0,-0.1197,0.425]
      TCP   offset = [0,0,0.39225]
    Input:
      p = (x,y,z)    target TCP position
      r = (rx,ry,rz) target TCP orientation in URDF RPY (roll‑x, pitch‑y, yaw‑z)
    Returns:
      (q1, q2, q3) in radians
    """
    px, py, pz = p
    rx, ry, rz = r
    cx = math.cos(rx)
    if cx >= 0.0:
        theta_sum = ry
        q1_init = rz
    else:
        theta_sum = math.pi - ry
        q1_init = rz + math.pi
    q1 = math.atan2(math.sin(q1_init), math.cos(q1_init))
    c1 = math.cos(q1)
    s1 = math.sin(q1)
    x1 = c1 * px + s1 * py
    z1 = pz
    b = 0.39225
    x2 = x1 - b * math.sin(theta_sum)
    z2 = z1 - b * math.cos(theta_sum)
    a = 0.425
    q2 = math.atan2(x2, z2)
    q3 = theta_sum - q2
    if cx < 0.0:
        q3 = math.atan2(math.sin(q3), math.cos(q3))
        q2 = theta_sum - q3
    return (q1, q2, q3)