def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach TCP position "p".
    
    The robot has four joints with the following configuration (with link offsets derived from the URDF):
      - Revolute 1 (rotation about Y): Base rotation.
      - Revolute 2 (rotation about Y): Offset d1 = [0, -0.1197, 0.425].
      - Revolute 3 (rotation about Y): Offset d2 = [0, 0, 0.39225].
      - Revolute 4 (rotation about Z): Offset d3 = [0, 0.093, 0] and TCP offset d4 = [0, 0, 0.09465].
      
    Observations:
      - Rotations about Y leave the Y-coordinate unchanged. The total fixed Y offset is:
           -0.1197 (from d1's y) + 0.093 (from d3's y) = -0.0267.
         Thus, the target p[1] is expected to be -0.0267.
      - The fourth joint is redundant for positioning; we choose q4 = 0.
      
    Strategy:
      1. Choose the base rotation (q1) such that when “undoing” the base rotation, the TCP lies in the x=0 plane.
         This can be achieved by: q1 = arctan2(x, z).
      2. Compute the distance in the x-z plane as:
            r = sqrt(x² + z²)
         Note: the fixed offset along z (0.425 from d1) must be subtracted to obtain the effective length for the planar chain.
      3. For the planar 2R chain formed by joints 2 and 3:
           Let a = 0.09465 (from d4, TCP offset) and b = 0.39225 (from d2).
         The effective arm length is:
           R_arm = r - 0.425.
         Using the cosine law, we obtain:
           cos(q3) = (R_arm² - (a² + b²)) / (2 * a * b).
         To avoid domain errors from floating point inaccuracies, clamp cos(q3) within [-1, 1].
      4. Once q3 is computed, compute:
           phi = arctan2(a*sin(q3), b + a*cos(q3))
         and set:
           q2 = -phi.
      5. Set the redundant joint:
           q4 = 0.
    
    :param p: The TCP position as a tuple (x, y, z). (The y-coordinate is expected to be -0.0267.)
    :return: A tuple (q1, q2, q3, q4) representing the joint angles in radians.
    """
    import math
    x, y, z = p
    q1 = math.atan2(x, z)
    r = math.sqrt(x ** 2 + z ** 2)
    d1_z = 0.425
    a = 0.09465
    b = 0.39225
    R_arm = r - d1_z
    cos_q3 = (R_arm ** 2 - (a ** 2 + b ** 2)) / (2 * a * b)
    cos_q3 = max(min(cos_q3, 1.0), -1.0)
    q3 = math.acos(cos_q3)
    phi = math.atan2(a * math.sin(q3), b + a * math.cos(q3))
    q2 = -phi
    q4 = 0.0
    return (q1, q2, q3, q4)