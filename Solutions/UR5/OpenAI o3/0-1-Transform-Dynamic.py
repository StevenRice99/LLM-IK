import math
from typing import Tuple

def _wrap(a: float) -> float:
    """
    Wrap an angle to the interval [-π,\xa0π].
    """
    return math.atan2(math.sin(a), math.cos(a))

def inverse_kinematics(p: Tuple[float, float, float], r: Tuple[float, float, float]) -> Tuple[float, float]:
    """
    Closed‑form inverse kinematics for the 2‑DoF chain described in the
    DETAILS section.

    Forward kinematics orientation:
        R = Rz(θ1) · Ry(θ2)

    URDF (extrinsic) RPY convention:
        R = Rz(yaw) · Ry(pitch) · Rx(roll)

    Matching the two matrices gives two admissible cases that the mechanism can
    realise:

      ┌────────────────────────────┬────────────────────────────────┐
      │        TCP RPY             │      Joint solution            │
      ├────────────────────────────┼────────────────────────────────┤
      │ |roll|   < π/2   (≈ 0)     │ θ1 =  yaw                      │
      │                            │ θ2 =  pitch                    │
      ├────────────────────────────┼────────────────────────────────┤
      │ |roll| ≥ π/2   (≈ ±π)      │ θ1 =  yaw + π                  │
      │                            │ θ2 =  π   - pitch              │
      └────────────────────────────┴────────────────────────────────┘

    (both rows are finally wrapped to [-π,\xa0π]).

    Parameters
    ----------
    p : (x,\xa0y,\xa0z)  – TCP position (not required for this manipulator but kept
        for compatibility with the requested signature).
    r : (roll,\xa0pitch,\xa0yaw) – Desired TCP orientation in radians.

    Returns
    -------
    (θ1,\xa0θ2) – Joint angles in radians, wrapped to [-π,\xa0π].
    """
    roll, pitch, yaw = r
    roll_wrapped = _wrap(roll)
    if abs(roll_wrapped) < math.pi / 2:
        theta1 = yaw
        theta2 = pitch
    else:
        theta1 = yaw + math.pi
        theta2 = math.pi - pitch
    return (_wrap(theta1), _wrap(theta2))