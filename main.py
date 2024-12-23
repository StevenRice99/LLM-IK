import copy
import logging
import os.path
import warnings

import ikpy.chain
import ikpy.utils.plot as plot_utils
import numpy as np
from ikpy.link import URDFLink
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from tabulate import tabulate

# Set up logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


class Robot:
    """
    Handle all aspects of serial robots.
    """

    def __init__(self, path: str):
        """
        Initialize the robot.
        :param path: The path to the URDF to load.
        """
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.chains = None
        self.joints = 0
        # Nothing to do if the file does not exist.
        if not os.path.exists(path):
            logging.error(f"{self.name} | Path '{path}' does not exist.")
            return
        # Nothing to do if a directory was passed.
        if not os.path.isfile(path):
            logging.error(f"{self.name} | Path '{path}' is not a file.")
            return
        # Try to load the file, exiting if there are errors.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                chain = ikpy.chain.Chain.from_urdf_file(path)
        except Exception as e:
            logging.error(f"{self.name} | Could not parse '{path}': {e}")
            return
        # Get the joints we need, as any leading fixed ones can be removed.
        zeros = np.array([0, 0, 0])
        links = []
        active = []
        for link in chain.links:
            # We only care for properly formatted links.
            if not isinstance(link, URDFLink):
                continue
            # If this is not a fixed link, it is a joint we can set.
            joint = link.joint_type != "fixed"
            active.append(joint)
            if joint:
                self.joints += 1
            # If this is the first properly formatted link, it should be at the origin.
            if len(links) > 0:
                origin_translation = link.origin_translation
                origin_orientation = link.origin_orientation
            # Otherwise, use the existing offset.
            else:
                origin_translation = zeros
                origin_orientation = zeros
            # Store the link.
            links.append(URDFLink(f"{len(links) + 1}", origin_translation, origin_orientation, link.rotation,
                                  link.translation, link.bounds, "rpy", link.use_symbolic_matrix, link.joint_type))
        if self.joints < 1:
            logging.error(f"{self.name} | No joints.")
            return
        # Build a lookup for the moveable joints in the links.
        total = len(links)
        joint_indices = {}
        index = 0
        for i in range(total):
            if active[i]:
                joint_indices[index] = i
                index += 1
        # Build all possible sub chains.
        self.chains = {}
        # All possible starting lower joints.
        for lower in range(self.joints):
            self.chains[lower] = {}
            # All possible upper ending joints.
            for upper in range(lower, self.joints):
                # Get the starting and ending joint indices.
                lower_index = joint_indices[lower]
                upper_index = joint_indices[upper] + 1
                # If the ending joint is the last in the chain, use the whole chain.
                if upper_index >= total:
                    instance_links = links[lower_index:]
                    instance_active = active[lower_index:]
                # Otherwise, use the active joints, but set the next out of bounds joint as an inactive end effector.
                else:
                    instance_links = links[lower_index:upper_index]
                    instance_active = active[lower_index:upper_index]
                    tcp = links[upper_index]
                    instance_links.append(URDFLink("TCP", tcp.origin_translation, tcp.origin_orientation,
                                                   use_symbolic_matrix=tcp.use_symbolic_matrix, joint_type="fixed"))
                    instance_active.append(False)
                # Ensure the base joint is at the origin.
                base = instance_links[0]
                instance_links[0] = URDFLink("1", zeros, zeros, base.rotation, base.translation, base.bounds,
                                             "rpy", base.use_symbolic_matrix, base.joint_type)
                # Break references to ensure we can name each link properly.
                instance_links = copy.deepcopy(instance_links)
                instance_count = len(instance_links) - 1
                for i in range(1, instance_count):
                    instance_links[i].name = f"{i + 1}"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    self.chains[lower][upper] = ikpy.chain.Chain(instance_links, instance_active,
                                                                 f"{self.name}-{lower}-{upper}")
        logging.info(f"{self.name} | Loaded | {self.joints} Joints")

    def __str__(self) -> str:
        return self.details()

    def details(self, lower: int = 0, upper: int = -1, limits: bool = True) -> str:
        if not self.is_valid():
            logging.error(f"{self.name} | Details | Robot not configured.")
            return ""
        # Get all values to perform.
        lower, upper = self.validate_lower_upper(lower, upper)
        chain = self.chains[lower][upper]
        total = len(chain.links)
        headers = ["Link", "Type", "Position", "Orientation"]
        have_orientation = False
        have_translation = False
        have_limits = False
        for i in range(total):
            if chain.links[i].rotation is not None:
                have_orientation = True
            if chain.links[i].translation is not None:
                have_translation = True
            if chain.links[i].bounds is not None:
                have_limits = True
            if have_orientation and have_translation and have_limits:
                break
        if have_orientation:
            headers.append("Axes")
        if have_translation:
            headers.append("Translation")
        if limits and have_limits:
            headers.append("Limits")
        data = []
        for i in range(total):
            link = chain.links[i]
            details = [link.name, link.joint_type.capitalize(),
                       f"[{neat(link.origin_translation[0])}, {neat(link.origin_translation[1])}, "
                       f"{neat(link.origin_translation[2])}]",
                       f"[{neat(link.origin_orientation[0])}, {neat(link.origin_orientation[1])}, "
                       f"{neat(link.origin_orientation[2])}]"]
            if have_orientation:
                if link.rotation is None:
                    details.append("-")
                else:
                    details.append(f"[{neat(link.rotation[0])}, {neat(link.rotation[1])}, {neat(link.rotation[2])}]")
            if have_translation:
                if link.translation is None:
                    details.append("-")
                else:
                    details.append(f"[{neat(link.translation[0])}, {neat(link.translation[1])}, "
                                   f"{neat(link.translation[2])}]")
            if limits and have_limits:
                if link.joint_type == "fixed" or link.bounds is None:
                    details.append("-")
                else:
                    details.append(f"[{neat(link.bounds[0])}, {neat(link.bounds[1])}]")
            data.append(details)
        return tabulate(data, headers, tablefmt="presto")

    def forward_kinematics(self, lower: int = 0, upper: int = -1, joints: list[float] or None = None,
                           plot: bool = False, width: float = 10, height: float = 10,
                           target: list[float] or None = None) -> (list[float], list[float]):
        """
        Perform forward kinematics.
        :param lower: The starting joint.
        :param upper: The ending joint.
        :param joints: Values to set the joints to.
        :param plot: If the result should be plotted.
        :param width: The width to plot.
        :param height: The height to plot.
        :param target: The target to display in the plot.
        :return: The position and orientation from the forward kinematics.
        """
        # Ensure we can perform forward kinematics.
        if not self.is_valid():
            logging.error(f"{self.name} | Forward Kinematics | Robot not configured.")
            return [0, 0, 0], [0, 0, 0]
        # Get all values to perform.
        lower, upper = self.validate_lower_upper(lower, upper)
        chain = self.chains[lower][upper]
        total = len(chain.links)
        # Set the joints.
        values = [0] * total
        index = 0
        last = 0 if joints is None else len(joints)
        controlled = []
        for i in range(total):
            # Keep fixed joints at zero.
            if chain.links[i].joint_type == "fixed":
                continue
            # If joint values were passed, set the joint value.
            if index < last:
                values[i] = joints[index]
                index += 1
            # Otherwise, if not passed and there are bounds, set the midpoint.
            elif chain.links[i].bounds is not None:
                values[i] = np.average(chain.links[i].bounds)
            controlled.append(values[i])
        # Perform forward kinematics.
        forward = chain.forward_kinematics(values)
        # Get the position and orientation reached.
        position = forward[:3, 3]
        orientation = Rotation.from_matrix(forward[:3, :3]).as_euler("xyz")
        # Plot if we should.
        if plot:
            fig, ax = plot_utils.init_3d_figure()
            fig.set_size_inches(abs(width), abs(height))
            # If a target to display was passed, display it, otherwise display the end position.
            chain.plot(values, ax, position if target is None else target)
            plt.title(f"{self.name} from joints {lower + 1} to {upper + 1}")
            plt.tight_layout()
            plt.show()
        logging.debug(f"{self.name} | {lower + 1} to {upper + 1} | Forward kinematics | Joints = {controlled} | "
                      f"Position = {position} | Orientation = {orientation}")
        return position, orientation

    def inverse_kinematics(self, lower: int = 0, upper: int = -1, position: list[float] or None = None,
                           orientation: list[float] or None = None, plot: bool = False, width: float = 10,
                           height: float = 10) -> (list[float], float, float):
        """
        Perform inverse kinematics.
        :param lower: The starting joint.
        :param upper: The ending joint.
        :param position: The position to solve for.
        :param orientation: The orientation to solve for.
        :param plot: If the result should be plotted.
        :param width: The width to plot.
        :param height: The height to plot.
        :return: The solution joints, positional error, and rotational error.
        """
        # Ensure we can perform inverse kinematics.
        if not self.is_valid():
            logging.error(f"{self.name} | Inverse Kinematics | Robot not configured.")
            return [], np.inf, np.inf
        if position is None:
            logging.warning(f"{self.name} | Inverse Kinematics | No target position was passed, solving for [0, 0, 0].")
            position = [0, 0, 0]
        # Set the joints to start at the midpoints.
        lower, upper = self.validate_lower_upper(lower, upper)
        chain = self.chains[lower][upper]
        total = len(chain.links)
        values = [0] * total
        for i in range(total):
            if chain.links[i].joint_type != "fixed" and chain.links[i].bounds is not None:
                values[i] = np.average(chain.links[i].bounds)
        # Set the target pose.
        target = np.eye(4)
        if orientation is not None:
            target[:3, :3] = Rotation.from_euler("xyz", orientation).as_matrix()
        target[:3, 3] = position
        # Solve the inverse kinematics.
        values = chain.inverse_kinematics_frame(target, orientation_mode=None if orientation is None else "all",
                                                initial_position=values)
        # Get the actual joint values, ignoring fixed links.
        parsed = []
        for i in range(total):
            if chain.links[i].joint_type != "fixed":
                parsed.append(values[i])
        # Get the reached position and orientations.
        true_position, true_orientation = self.forward_kinematics(lower, upper, parsed)
        # Get the position error.
        distance = np.sqrt(sum([(goal - true) ** 2 for goal, true in zip(position, true_position)]))
        # Get the orientation error if it was being solved for.
        if orientation is None:
            angle = 0
        else:
            angle = sum([abs(goal - true) for goal, true in zip(orientation, true_orientation)])
        # Plot if we should.
        if plot:
            fig, ax = plot_utils.init_3d_figure()
            fig.set_size_inches(abs(width), abs(height))
            # Show the goal position that should have been reached.
            chain.plot(values, ax, position)
            plt.title(f"{self.name} from joints {lower + 1} to {upper + 1}")
            plt.tight_layout()
            plt.show()
        if orientation:
            logging.debug(f"{self.name} | {lower + 1} to {upper + 1} | Inverse Kinematics | Target Position = "
                          f"{position} | Reached Position = {true_position} | Position Error = {distance} | Target "
                          f"Orientation = {orientation} | Reached Orientation = {true_orientation} | Orientation Error "
                          f"= {angle} | Solution = {parsed}")
        else:
            logging.debug(f"{self.name} | {lower + 1} to {upper + 1} | Inverse Kinematics | Target = {position} | "
                          f"Reached = {true_position} | Error = {distance} | Solution = {parsed}")
        return parsed, distance, angle

    def is_valid(self) -> bool:
        """
        Ensure the robot is valid.
        :return: True if the robot is valid, false otherwise.
        """
        return self.chains is not None

    def validate_lower_upper(self, lower: int = 0, upper: int = -1) -> (int, int):
        """
        Validate the lower and upper joints to perform on.
        :param lower:
        :param upper:
        :return:
        """
        # If no upper value was passed, use the last joint.
        if upper < 0:
            upper = self.joints - 1
        # If no lower value was passed, use the first joint.
        if lower < 0:
            lower = 0
        # Ensure the lower value is at most equal to the upper value for a single joint chain.
        elif lower > upper:
            lower = upper
        # Return the updated lower and upper values.
        return lower, upper


def reached(distance: float = 0, angle: float = 0, distance_error: float = 0.001, angle_error: float = 0.001):
    """
    Check if a robot has reached a target
    :param distance: The distance from the target.
    :param angle: The angle from the target.
    :param distance_error: The maximum acceptable positional error.
    :param angle_error: The maximum acceptable orientation error.
    :return:
    """
    return distance <= distance_error and angle <= angle_error


def neat(value: float) -> str:
    """
    Format a float value with no trailing zeros.
    :param value: The float value.
    :return: The value as a formatted string.
    """
    return f"{value:.8f}".rstrip('0').rstrip('.')


def main() -> None:
    """
    Handle main program operations.
    :return: Nothing.
    """
    ur5 = Robot("ur5.urdf")
    # ur5.forward_kinematics(plot=True)
    # j, d, a = ur5.inverse_kinematics(plot=True)
    # print(j)
    # print(d)
    # print(a)
    # print(reached(d, a))
    print(ur5)


if __name__ == "__main__":
    main()
