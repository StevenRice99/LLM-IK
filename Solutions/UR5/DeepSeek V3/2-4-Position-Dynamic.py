import math

def forward_kinematics(theta1: float, theta2: float, theta3: float) -> tuple[float, float, float]:
    """
    Computes the TCP position given the joint angles.
    :param theta1: Joint 1 angle in radians.
    :param theta2: Joint 2 angle in radians.
    :param theta3: Joint 3 angle in radians.
    :return: The TCP position in the form [x, y, z].
    """
    # Link lengths
    L1 = 0.39225
    L2 = 0.093
    L3 = 0.09465
    
    # Transformation matrices
    # Joint 1: Rotation around Y-axis
    T1 = [
        [math.cos(theta1), 0, math.sin(theta1), 0],
        [0, 1, 0, 0],
        [-math.sin(theta1), 0, math.cos(theta1), L1],
        [0, 0, 0, 1]
    ]
    
    # Joint 2: Rotation around Y-axis
    T2 = [
        [math.cos(theta2), 0, math.sin(theta2), 0],
        [0, 1, 0, 0],
        [-math.sin(theta2), 0, math.cos(theta2), L2],
        [0, 0, 0, 1]
    ]
    
    # Joint 3: Rotation around Z-axis
    T3 = [
        [math.cos(theta3), -math.sin(theta3), 0, 0],
        [math.sin(theta3), math.cos(theta3), 0, 0],
        [0, 0, 1, L3],
        [0, 0, 0, 1]
    ]
    
    # Combined transformation
    T = [
        [T1[0][0]*T2[0][0]*T3[0][0] + T1[0][1]*T2[1][0]*T3[0][0] + T1[0][2]*T2[2][0]*T3[0][0],
        T1[0][0]*T2[0][1]*T3[0][1] + T1[0][1]*T2[1][1]*T3[0][1] + T1[0][2]*T2[2][1]*T3[0][1],
        T1[0][0]*T2[0][2]*T3[0][2] + T1[0][1]*T2[1][2]*T3[0][2] + T1[0][2]*T2[2][2]*T3[0][2],
        T1[0][0]*T2[0][3]*T3[0][3] + T1[0][1]*T2[1][3]*T3[0][3] + T1[0][2]*T2[2][3]*T3[0][3]
    ]
    
    x = T[0][3]
    y = T[1][3]
    z = T[2][3]
    
    return (x, y, z)

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    
    # Link lengths
    L1 = 0.39225
    L2 = 0.093
    L3 = 0.09465
    
    # Solve for θ3
    theta3 = math.atan2(x, z)
    
    # Solve for θ2
    r = math.sqrt(x**2 + z**2)
    D = (r**2 + y**2 - L1**2 - L2**2 - L3**2) / (2 * L2 * L3)
    theta2 = math.atan2(math.sqrt(1 - D**2), D)
    
    # Solve for θ1
    theta1 = math.atan2(y, r) - math.atan2(L3 * math.sin(theta2), L2 + L3 * math.cos(theta2))
    
    return (theta1, theta2, theta3)