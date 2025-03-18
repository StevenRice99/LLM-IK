import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    px, py, pz = p
    rx, ry, rz = r
    L1 = 0.39225
    L2 = 0.093
    L3 = 0.09465
    pz_eff = pz - L3
    d = np.sqrt(px ** 2 + py ** 2)
    θ2 = np.arctan2(pz_eff - L1, d)
    θ1 = np.arctan2(py, px)
    θ3 = rz
    return (θ1, θ2, θ3)