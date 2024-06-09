import numpy as np


def inverse_kinematics(p: list, joint_limit=None) -> list:
  """
  Solve the inverse kinematics to reach the position in the reference frame of the chain's origin.

  :param p: The position for the end effector to reach in a Cartesian [x, y, z].
  :param joint_limit: Optional limit for the revolute joint (in radians).
  :return: A list containing the closest reachable joint values in radians,
           or None if not reachable within joint limits.
  """
  # Check if desired Z position is reachable by the arm
  if p[2] < 0 or p[2] > 0.4:
    return None  # Z is out of reach

  # Target end-effector position in relation to Joint 1
  target_pos = [p[0], p[1], 0]

  # Get the distance between target and base (radius)
  radius = np.linalg.norm(target_pos)

  # Apply arctangent function to find desired angles (considering both signs)
  theta1_1 = np.arctan2(target_pos[1], target_pos[0])
  theta1_2 = np.pi + theta1_1

  # Apply joint limits if provided
  if joint_limit is not None:
    theta1_1 = max(min(theta1_1, joint_limit), -joint_limit)
    theta1_2 = max(min(theta1_2, joint_limit), -joint_limit)

  # Select solution closer to desired Z
  if abs(0.4 - p[2]) < abs(0.4 - np.cos(theta1_1)):
    return [theta1_1]
  else:
    return [theta1_2]
