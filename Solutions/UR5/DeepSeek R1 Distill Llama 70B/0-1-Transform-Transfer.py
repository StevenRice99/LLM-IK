import math
from scipy.spatial.transform import Rotation

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the joints to for reaching position "p" and orientation "r".
    """
    rot = Rotation.from_euler('xyz', r, degrees=False)
    R_desired = rot.as_matrix()
    sinθ2 = -R_desired[2, 0]
    cosθ2 = R_desired[2, 2]
    θ2 = math.atan2(sinθ2, cosθ2)
    sinθ1 = -R_desired[0, 1]
    cosθ1 = R_desired[1, 1]
    θ1 = math.atan2(sinθ1, cosθ1)
    x = -0.13585 * math.sin(θ1) * math.cos(θ2) + 0.425 * math.sin(θ2)
    y = 0.13585 * math.cos(θ1) - 0.1197
    z = 0.13585 * math.sin(θ1) * math.sin(θ2) + 0.425 * math.cos(θ2)
    return (θ1, θ2)