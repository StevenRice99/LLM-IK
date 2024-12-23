import logging
import os.path
import warnings

import ikpy.chain
import ikpy.utils.plot as plot_utils
import numpy as np
from ikpy.link import URDFLink
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

# Set up logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


class Robot:
    def __init__(self, path: str):
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.chains = {}
        if not os.path.exists(path):
            logging.error(f"{self.name} | Path '{path}' does not exist.")
            return
        if not os.path.isfile(path):
            logging.error(f"{self.name} | Path '{path}' is not a file.")
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                chain = ikpy.chain.Chain.from_urdf_file(path)
        except Exception as e:
            logging.error(f"{self.name} | Could not parse '{path}': {e}")
            return
        self.joints = 0
        links = []
        active = []
        for link in chain.links:
            if isinstance(link, URDFLink):
                if len(links) > 0:
                    origin_translation = link.origin_translation
                    origin_orientation = link.origin_orientation
                else:
                    origin_translation = [0, 0, 0]
                    origin_orientation = [0, 0, 0]
                links.append(URDFLink(link.name, origin_translation, origin_orientation, link.rotation,
                                      link.translation, link.bounds, "rpy", link.use_symbolic_matrix, link.joint_type))
                joint = link.joint_type != "fixed"
                active.append(joint)
                if joint:
                    self.joints += 1
        total = len(links)
        joint_indices = {}
        index = 0
        for i in range(total):
            if active[i]:
                joint_indices[index] = i
                index += 1
        for lower in range(self.joints):
            self.chains[lower] = {}
            for upper in range(lower, self.joints):
                lower_index = joint_indices[lower]
                upper_index = joint_indices[upper] + 1
                if upper_index >= total:
                    instance_links = links[lower_index:]
                    instance_active = active[lower_index:]
                else:
                    instance_links = links[lower_index:upper_index]
                    instance_active = active[lower_index:upper_index]
                    tcp = links[upper_index]
                    instance_links.append(URDFLink(tcp.name, tcp.origin_translation, tcp.origin_orientation,
                                                   use_symbolic_matrix=tcp.use_symbolic_matrix, joint_type="fixed"))
                    instance_active.append(False)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    self.chains[lower][upper] = ikpy.chain.Chain(instance_links, instance_active,
                                                                 f"{self.name}-{lower}-{upper}")

    def forward_kinematics(self, lower: int = 0, upper: int = -1, joints: list[float] or None = None,
                           plot: bool = False, width: float = 10, height: float = 10,
                           target: list[float] or None = None) -> (list[float], list[float]):
        lower, upper = self.validate_lower_upper(lower, upper)
        chain = self.chains[lower][upper]
        total = len(chain.links)
        values = [0] * total
        index = 0
        last = 0 if joints is None else len(joints)
        for i in range(total):
            if chain.links[i].joint_type != "fixed":
                if index < last:
                    values[i] = joints[index]
                    index += 1
                elif chain.links[i].bounds is not None:
                    values[i] = np.average(chain.links[i].bounds)
        forward = chain.forward_kinematics(values)
        position = forward[:3, 3]
        orientation = Rotation.from_matrix(forward[:3, :3]).as_euler("xyz")
        if plot:
            fig, ax = plot_utils.init_3d_figure()
            fig.set_size_inches(abs(width), abs(height))
            chain.plot(values, ax, position if target is None else target)
            plt.title(f"{self.name} from joints {lower + 1} to {upper + 1}")
            plt.tight_layout()
            plt.show()
        return position, orientation

    def inverse_kinematics(self, lower: int = 0, upper: int = -1, position: list[float] or None = None,
                           orientation: list[float] or None = None, plot: bool = False, width: float = 10,
                           height: float = 10) -> (list[float], float, float):
        if position is None:
            position = [0, 0, 0]
        lower, upper = self.validate_lower_upper(lower, upper)
        chain = self.chains[lower][upper]
        total = len(chain.links)
        values = [0] * total
        for i in range(total):
            if chain.links[i].joint_type != "fixed" and chain.links[i].bounds is not None:
                values[i] = np.average(chain.links[i].bounds)
        target = np.eye(4)
        if orientation is not None:
            target[:3, :3] = Rotation.from_euler("xyz", orientation).as_matrix()
        target[:3, 3] = position
        values = chain.inverse_kinematics_frame(target, orientation_mode="all", initial_position=values)
        parsed = []
        for i in range(total):
            if chain.links[i].joint_type != "fixed":
                parsed.append(values[i])
        true_position, true_orientation = self.forward_kinematics(lower, upper, parsed)
        distance = np.sqrt(sum([(goal - true) ** 2 for goal, true in zip(position, true_position)]))
        if orientation is None:
            angle = 0
        else:
            angle = sum([abs(goal - true) for goal, true in zip(orientation, true_orientation)])
        if plot:
            fig, ax = plot_utils.init_3d_figure()
            fig.set_size_inches(abs(width), abs(height))
            chain.plot(values, ax, position)
            plt.title(f"{self.name} from joints {lower + 1} to {upper + 1}")
            plt.tight_layout()
            plt.show()
        return parsed, distance, angle

    def validate_lower_upper(self, lower: int = 0, upper: int = -1) -> (int, int):
        if upper < 0:
            upper = self.joints - 1
        if lower < 0:
            lower = 0
        elif lower > upper:
            lower = upper
        return lower, upper


def reached(distance: float = 0, angle: float = 0, distance_error: float = 0.001, angle_error: float = 0.001):
    return distance <= distance_error and angle <= angle_error


if __name__ == "__main__":
    ur5 = Robot("ur5.urdf")
    #ur5.forward_kinematics(plot=True)
    j, d, a = ur5.inverse_kinematics(plot=True)
    print(j)
    print(d)
    print(a)
    print(reached(d, a))
