def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Computes the joint angles (theta1, theta2, theta3) for a 3-dof manipulator whose forward kinematics are
      TCP = Rz(theta1) * { [0, 0.13585, 0] +
                          Ry(theta2)*[0, -0.1197, 0.425] +
                          Ry(theta2+theta3)*[0, 0, 0.39225] }.
    
    The tool’s desired pose is given by position p and Euler angles r = (roll, pitch, yaw) as defined by URDF,
    i.e. the orientation matrix is built as R_target = Rz(yaw)*Ry(pitch)*Rx(roll). However, because the manipulator
    produces only rotations of the form Rz(theta1)*Ry(theta2+theta3), we must “extract” effective orientation parameters 
    (theta1 and phi with phi = theta2+theta3) from the target R_target.
    
    The procedure is as follows:
      1. Compute R_target = Rz(yaw)*Ry(pitch)*Rx(roll) using the standard formulas.
      2. Because any rotation produced by the manipulator is of the form:
             R = Rz(theta1)*Ry(phi)
         which explicitly is:
             [[ cos(theta1)*cos(phi), -sin(theta1), cos(theta1)*sin(phi)],
              [ sin(theta1)*cos(phi),  cos(theta1), sin(theta1)*sin(phi)],
              [       -sin(phi)    ,      0     ,     cos(phi)     ]],
         we can “read” phi and theta1 from R_target:
             phi   = atan2( -R_target[2,0], R_target[2,2] )
             theta1 = atan2( -R_target[0,1], R_target[1,1] )
      3. Rotate the target position p into the frame that “undoes” the base rotation Rz(theta1):
             p' = Rz(-theta1) * p.
         In that frame the structure of the translations is:
             p'_x = 0.425*sin(theta2) + 0.39225*sin(phi)
             p'_z = 0.425*cos(theta2) + 0.39225*cos(phi)
         so that one may solve for theta2 by:
             theta2 = atan2( p'_x - 0.39225*sin(phi),  p'_z - 0.39225*cos(phi) )
         and then theta3 = phi - theta2.
      
    This method automatically “corrects” for cases in which the desired TCP rotation (given as r=[roll, pitch, yaw])
    may have a roll of ±pi, so that the achievable rotation Rz(theta1)*Ry(theta2+theta3) matches R_target.
    
    :param p: (px, py, pz) desired TCP position.
    :param r: (roll, pitch, yaw) desired TCP orientation in radians (URDF rpy convention: R_target = Rz(yaw)*Ry(pitch)*Rx(roll)).
    :return: A tuple (theta1, theta2, theta3) that produces the target pose.
    """
    import math
    px, py, pz = p
    roll, pitch, yaw = r
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    A = [[0, 0, 0] for _ in range(3)]
    A[0][0] = cp
    A[0][1] = -sp * sr
    A[0][2] = sp * cr
    A[1][0] = 0
    A[1][1] = cr
    A[1][2] = -sr
    A[2][0] = -sp
    A[2][1] = -cp * sr
    A[2][2] = cp * cr
    R_target = [[0, 0, 0] for _ in range(3)]
    for i in range(3):
        R_target[i][0] = cy * A[0][0] + -sy * A[1][0] + 0 * A[2][0]
        R_target[i][1] = cy * A[0][1] + -sy * A[1][1] + 0 * A[2][1]
        R_target[i][2] = cy * A[0][2] + -sy * A[1][2] + 0 * A[2][2]
    R_target[2][0] = A[2][0]
    R_target[2][1] = A[2][1]
    R_target[2][2] = A[2][2]
    R_target[0][0] = cy * A[0][0] + -sy * A[1][0]
    R_target[0][1] = cy * A[0][1] + -sy * A[1][1]
    R_target[0][2] = cy * A[0][2] + -sy * A[1][2]
    R_target[1][0] = sy * A[0][0] + cy * A[1][0]
    R_target[1][1] = sy * A[0][1] + cy * A[1][1]
    R_target[1][2] = sy * A[0][2] + cy * A[1][2]
    phi = math.atan2(-R_target[2][0], R_target[2][2])
    theta1 = math.atan2(-R_target[0][1], R_target[1][1])
    cos_t1 = math.cos(theta1)
    sin_t1 = math.sin(theta1)
    p_prime_x = cos_t1 * px + sin_t1 * py
    p_prime_z = pz
    theta2 = math.atan2(p_prime_x - 0.39225 * math.sin(phi), p_prime_z - 0.39225 * math.cos(phi))
    theta3 = phi - theta2
    return (theta1, theta2, theta3)