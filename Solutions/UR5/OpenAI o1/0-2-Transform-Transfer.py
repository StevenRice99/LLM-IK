def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Analytical 3-DOF inverse kinematics for a serial chain with joints:
      1) Revolute about Z
      2) Revolute about Y
      3) Revolute about Y
    and link offsets (in their parent frames):
      Joint1: position [0, 0, 0]
      Joint2: position [0, 0.13585, 0]
      Joint3: position [0, -0.1197, 0.425]
      TCP:    position [0, 0, 0.39225]

    The manipulator can only realize a net orientation of Rz(θ1)*Ry(θ2)*Ry(θ3).
    In URDF notation (roll, pitch, yaw), we effectively have no roll DOF, so any non-zero
    roll can often be “absorbed” by shifting pitch/yaw. Below, we do a “zero-roll” extraction:

      1) From the input roll–pitch–yaw, build the 3×3 rotation matrix.
      2) Extract an equivalent rotation Rz(yaw_eff)*Ry(pitch_eff) that “ignores” roll
         by matching only the final Z and Y directions, discarding any real rotation
         about X that the arm can’t handle. This yields (θ1, θ2 + θ3) = (yaw_eff, pitch_eff).
      3) Solve the position p with the 2R geometry for joints 2,3, ensuring θ2 + θ3 = pitch_eff.

    :param p: The target TCP position (x, y, z).
    :param r: The target orientation in radians [roll, pitch, yaw].
    :return: (theta1, theta2, theta3) in radians, each within ±2π.
    """
    import math

    def Rx(a):
        ca, sa = (math.cos(a), math.sin(a))
        return [[1, 0, 0], [0, ca, -sa], [0, sa, ca]]

    def Ry(a):
        ca, sa = (math.cos(a), math.sin(a))
        return [[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]]

    def Rz(a):
        ca, sa = (math.cos(a), math.sin(a))
        return [[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]]

    def matmul(A, B):
        return [[sum((A[i][k] * B[k][j] for k in range(3))) for j in range(3)] for i in range(3)]

    def rpy_to_matrix(roll, pitch, yaw):
        return matmul(matmul(Rx(roll), Ry(pitch)), Rz(yaw))
    roll, pitch, yaw = r
    Rd = rpy_to_matrix(roll, pitch, yaw)
    eps = 1e-09
    r02 = Rd[0][2]
    r12 = Rd[1][2]
    r22 = Rd[2][2]
    sin_pitch_eff = math.sqrt(r02 * r02 + r12 * r12)
    cos_pitch_eff = r22
    if cos_pitch_eff > 1.0:
        cos_pitch_eff = 1.0
    if cos_pitch_eff < -1.0:
        cos_pitch_eff = -1.0
    base_pitch_eff = math.acos(cos_pitch_eff)
    pitch_eff = base_pitch_eff
    if abs(sin_pitch_eff) < eps:
        yaw_eff = math.atan2(Rd[1][0], Rd[0][0])
    else:
        yaw_eff = math.atan2(r12, r02)

    def wrap_pm_pi(a):
        return (a + math.pi) % (2 * math.pi) - math.pi
    yaw_eff = wrap_pm_pi(yaw_eff)
    pitch_eff = wrap_pm_pi(pitch_eff)
    x, y, z = p
    theta1 = yaw_eff
    cos1, sin1 = (math.cos(theta1), math.sin(theta1))
    x_loc = x * cos1 + y * sin1
    y_loc = -x * sin1 + y * cos1
    z_loc = z
    y_loc -= 0.13585
    Xp = x_loc - 0.39225 * math.sin(pitch_eff)
    Zp = z_loc - 0.39225 * math.cos(pitch_eff)
    L = 0.425
    denom = Xp * Xp + Zp * Zp
    if denom < 1e-12:
        theta2 = 0.0
    else:
        theta2 = math.atan2(Xp, Zp)
    theta3 = pitch_eff - theta2
    theta1 = wrap_pm_pi(theta1)
    theta2 = wrap_pm_pi(theta2)
    theta3 = wrap_pm_pi(theta3)
    return (theta1, theta2, theta3)