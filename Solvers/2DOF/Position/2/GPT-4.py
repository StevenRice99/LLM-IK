from math import atan2, sqrt, pi, acos


def inverse_kinematics(p: list) -> list[float]:
    x, y, z = p

    # Calculate the distance to the target point in the XY plane.
    r_xy = sqrt(x ** 2 + y ** 2)

    # Compute theta1 based on the XY projection.
    theta_1 = atan2(y, x)

    # Calculate the distance from the origin to the target in 3D space.
    r = sqrt(x ** 2 + y ** 2 + (z - 0.4) ** 2)

    # Adjust theta_2 to consider the physical offset along z-axis by the first joint and the full reach of the manipulator.
    if r > 0:
        # Law of Cosines to find the angle at the joint
        theta_2 = acos((0.4 ** 2 + r ** 2 - (z - 0.4) ** 2) / (2 * 0.4 * r))
    else:
        theta_2 = 0

    # Adjust theta_2 based on its quadrant and the actual height (z-value) of the target.
    if z < 0.4:
        theta_2 = -theta_2

    return [theta_1, theta_2]
