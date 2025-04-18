import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Closed‑form IK for the 3‑DOF Z–Y–Y arm.
    p = (x,y,z)    : desired TCP position
    r = (roll,pitch,yaw) in radians: desired TCP RPY
      – only roll=0 or ±π are reachable (in the RPY sense),
        and we fold the ±π case into a Z–Y solution.
    Returns (theta1, theta2, theta3) in radians.
    """
    x, y, z = p
    roll, pitch, yaw = r
    PI = math.pi
    TWO_PI = 2 * PI
    EPS = 1e-06
    if abs(roll) < EPS:
        yaw_eff = yaw
        phi = pitch
        wrap3 = False
        sub2π = False
    elif abs(abs(roll) - PI) < EPS:
        yaw_eff = yaw + PI
        phi = PI - pitch
        if roll < 0:
            wrap3 = True
            sub2π = False
        else:
            wrap3 = False
            sub2π = True
    else:
        yaw_eff = yaw
        phi = pitch
        wrap3 = False
        sub2π = False
    yaw_eff = (yaw_eff + PI) % TWO_PI - PI
    l2 = 0.425
    l4 = 0.39225
    c1 = math.cos(yaw_eff)
    s1 = math.sin(yaw_eff)
    x_p = c1 * x + s1 * y
    z_p = z
    sin2 = (x_p - l4 * math.sin(phi)) / l2
    cos2 = (z_p - l4 * math.cos(phi)) / l2
    sin2 = max(min(sin2, 1.0), -1.0)
    cos2 = max(min(cos2, 1.0), -1.0)
    theta2 = math.atan2(sin2, cos2)
    base3 = phi - theta2
    if wrap3:
        theta3 = (base3 + PI) % TWO_PI - PI
    elif sub2π and phi > PI:
        theta3 = base3 - TWO_PI
    else:
        theta3 = base3
    return (yaw_eff, theta2, theta3)