import argparse
import copy
import importlib
import importlib.util
import logging
import os.path
import random
import time
import traceback
import warnings

import ikpy.chain
import ikpy.utils.plot as plot_utils
import numpy as np
import pandas as pd
from ikpy.link import URDFLink
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from tabulate import tabulate

# Folders.
ROBOTS = "Robots"
MODELS = "Models"
PROVIDERS = "Providers"
KEYS = "Keys"
INFO = "Info"
INTERACTIONS = "Interactions"
SOLUTIONS = "Solutions"
RESULTS = "Results"

# If we should run the actual API calls or not.
RUN = False

# Execution modes.
NORMAL = "Normal"
EXTEND = "Extend"
DYNAMIC = "Dynamic"

# API interaction file naming.
MESSAGE = "Message"
RESPONSE = "Response"

# Data naming.
POSITION = "Position"
TRANSFORM = "Transform"
TRAINING_TITLE = "Training"
EVALUATING_TITLE = "Evaluating"

# Parameters.
TRAINING = 1
EVALUATING = 1
SEED = 42
FEEDBACKS = 0
EXAMPLES = 1
DISTANCE_ERROR = 0.001
ANGLE_ERROR = 0.001

# Default bounding value.
BOUND = 2 * np.pi


class Robot:
    """
    Handle all aspects of serial robots.
    """

    def __init__(self, name: str):
        """
        Initialize the robot.
        :param name: The name of the URDF to load.
        """
        # Make the name which was passed valid.
        if not name.endswith(".urdf"):
            name = name + ".urdf"
        self.name = os.path.splitext(name)[0]
        # Cache the to save info to.
        self.info = os.path.join(INFO, self.name)
        self.results = os.path.join(RESULTS, self.name, "IKPy")
        self.chains = None
        self.joints = 0
        self.training = 0
        self.evaluating = 0
        self.data = {}
        # Nothing to do if the file does not exist.
        path = os.path.join(ROBOTS, name)
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
            links.append(URDFLink(link.name, origin_translation, origin_orientation, link.rotation, link.translation,
                                  link.bounds, "rpy", link.use_symbolic_matrix, link.joint_type))
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
        os.makedirs(self.info, exist_ok=True)
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
                instance_links[0] = URDFLink(base.name, zeros, zeros, base.rotation, base.translation, base.bounds,
                                             "rpy", base.use_symbolic_matrix, base.joint_type)
                # Break references to ensure we can name each link properly.
                instance_links = copy.deepcopy(instance_links)
                # Name every joint based on its type.
                instance_count = len(instance_links)
                fixed = 0
                revolute = 0
                prismatic = 0
                for i in range(0, instance_count):
                    joint_type = instance_links[i].joint_type.capitalize()
                    if joint_type == "Fixed":
                        fixed += 1
                        number = fixed
                    elif joint_type == "Revolute":
                        revolute += 1
                        number = revolute
                    else:
                        prismatic += 1
                        number = prismatic
                    instance_links[i].name = f"{joint_type} {number}"
                # If the last joint is not fixed, this chain cannot be used.
                if instance_links[-1].joint_type != "fixed":
                    self.chains[lower][upper] = None
                    logging.warning(f"{self.name} | {lower + 1} to {upper + 1} | Last joint is not fixed; skipping.")
                    continue
                # Ensure the TCP is named correctly.
                instance_links[-1].name = "TCP"
                # Build and cache this sub chain.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    chain = ikpy.chain.Chain(instance_links, instance_active,
                                             f"{self.name} from joints {lower + 1} to {upper + 1}")
                    self.chains[lower][upper] = chain
                    with open(os.path.join(self.info, f"{lower + 1}-{upper + 1}.txt"), "w") as file:
                        file.write(self.details(lower, upper)[0])
        logging.info(f"{self.name} | Info saved to '{self.info}'.")
        # Set the seed for generating training and evaluating instances.
        self.load_data()

    def __str__(self) -> str:
        """
        Get the table of the details of the full robot.
        :return: The table of the details of the full robot.
        """
        return self.details()[0]

    def save_results(self) -> None:
        """
        Save the results for built-in inverse kinematics.
        :return: Nothing.
        """
        # Nothing to do if the robot is not valid.
        if not self.is_valid():
            logging.error(f"{self.name} | Save Results | Robot not configured.")
            return None
        # Loop for every link.
        for lower in range(self.joints):
            for upper in range(lower, self.joints):
                for orientation in [False, True]:
                    # Single joints only solve for position.
                    if orientation and lower == upper:
                        continue
                    # Get the data for this.
                    data = self.get_data(lower, upper, False, orientation)
                    total = len(data)
                    if total < 1:
                        continue
                    # Tabulate results.
                    successes = 0
                    total_distance = 0
                    total_angle = 0
                    total_time = 0
                    for point in data:
                        distance = point["Distance"]
                        angle = point["Angle"] if orientation else 0
                        total_time += point["Time"]
                        did_reach = reached(distance, angle)
                        if did_reach:
                            successes += 1
                            continue
                        total_distance += distance
                        total_angle += angle
                    # Format results.
                    failures = total - successes
                    total_distance = 0 if failures < 1 else neat(total_distance / failures)
                    total_angle = 0 if failures < 1 else neat(total_angle / failures)
                    total_time = neat(total_time / total)
                    successes = neat(successes / total * 100)
                    s = "Success Rate (%),Average Failure Distance"
                    if orientation:
                        s += ",Average Failure Angle (°)"
                    s += f",Elapsed Time (s)\n{successes}%,{total_distance}"
                    if orientation:
                        s += f",{total_angle}°"
                    s += f",{total_time} s"
                    # Save results.
                    os.makedirs(self.results, exist_ok=True)
                    path = os.path.join(self.results, f"{lower}-{upper}-{TRANSFORM if orientation else POSITION}.csv")
                    with open(path, "w") as file:
                        file.write(s)
        logging.info(f"{self.name} | Save Results | IKPy results saved to '{self.results}'.")

    def load_data(self) -> None:
        """
        Load data for the robot to use.
        :return: Nothing.
        """
        # Nothing to do if the robot is not valid.
        if not self.is_valid():
            logging.error(f"{self.name} | Load Data | Robot not configured.")
            return None
        # Clear any previous data.
        self.data = {}
        # Set the random seed.
        random.seed(SEED)
        np.random.seed(SEED)
        # If there is already data for this configuration, load it.
        path = os.path.join(self.info, f"{SEED}-{TRAINING}-{EVALUATING}.json")
        if os.path.exists(path):
            df = pd.read_json(path, orient="records", lines=True)
            self.data = df.to_dict(orient="dict")
            logging.info(f"{self.name}| Seed = {SEED} | Training = {TRAINING} | Evaluating = {EVALUATING} | Generated "
                         f"data loaded from '{path}'.")
            self.save_results()
            return None
        # Run all possible joint configurations.
        for lower in range(self.joints):
            self.data[lower] = {}
            for upper in range(lower, self.joints):
                bounds = []
                # Only run for valid chains.
                chain = self.chains[lower][upper]
                if chain is None:
                    continue
                # Define the bounds for randomly generating poses.
                for link in chain.links:
                    if link.joint_type == "fixed":
                        continue
                    if link.bounds is None or link.bounds == (-np.inf, np.inf):
                        bounds.append(None)
                    else:
                        bounds.append(link.bounds)
                # Create the data structures to hold the data.
                training_position = {"Joints": [], "Position": []}
                training_transform = None if lower == upper else {"Joints": [], "Position": [], "Orientation": []}
                evaluating_position = {"Position": [], "Distance": [], "Angle": [], "Time": []}
                evaluating_transform = None if lower == upper else {"Position": [], "Orientation": [], "Distance": [],
                                                                    "Angle": [], "Time": []}
                # Create the training and evaluation data.
                for part in [TRAINING_TITLE, EVALUATING_TITLE]:
                    # Run for the number of instances.
                    instances = TRAINING if part == TRAINING_TITLE else EVALUATING
                    for i in range(instances):
                        # Define random joint values.
                        joints = []
                        for bound in bounds:
                            if bound is None:
                                joints.append(np.random.uniform(-BOUND, BOUND))
                            else:
                                joints.append(np.random.uniform(bound[0], bound[1]))
                        # Perform a common forwards and inverse kinematics that both sets of data use.
                        positions, orientations = self.forward_kinematics(lower, upper, joints)
                        position = positions[-1]
                        orientation = orientations[-1]
                        joints, distance, angle, elapsed = self.inverse_kinematics(lower, upper, position)
                        # Build training data.
                        if part == TRAINING_TITLE:
                            # Get the position only inverse kinematics pose.
                            positions, orientations = self.forward_kinematics(lower, upper, joints)
                            training_position["Joints"].append(joints)
                            training_position["Position"].append(positions[-1])
                            # Get the transform inverse kinematics pose.
                            if training_transform is not None:
                                joints, distance, angle, elapsed = self.inverse_kinematics(lower, upper, position,
                                                                                           orientation)
                                positions, orientations = self.forward_kinematics(lower, upper, joints)
                                training_transform["Joints"].append(joints)
                                training_transform["Position"].append(positions[-1])
                                training_transform["Orientation"].append(orientations[-1])
                            continue
                        # Build the evaluating data, starting with the performance of the position only results.
                        evaluating_position["Position"].append(position)
                        evaluating_position["Distance"].append(distance)
                        evaluating_position["Angle"].append(angle)
                        evaluating_position["Time"].append(elapsed)
                        # Save the transform results as well.
                        if evaluating_transform is not None:
                            joints, distance, angle, elapsed = self.inverse_kinematics(lower, upper, position,
                                                                                       orientation)
                            evaluating_transform["Position"].append(position)
                            evaluating_transform["Orientation"].append(orientation)
                            evaluating_transform["Distance"].append(distance)
                            evaluating_transform["Angle"].append(angle)
                            evaluating_transform["Time"].append(elapsed)
                # Cache the data.
                self.data[lower][upper] = {
                    TRAINING_TITLE: {POSITION: training_position, TRANSFORM: training_transform},
                    EVALUATING_TITLE: {POSITION: evaluating_position, TRANSFORM: evaluating_transform}
                }
                logging.info(f"{self.name} | {lower + 1} to {upper + 1} | Seed = {SEED} | Training = {TRAINING} | "
                             f"Evaluating = {EVALUATING} | Data generated.")
        # Save the newly generated data.
        os.makedirs(self.info, exist_ok=True)
        df = pd.DataFrame(self.data)
        df.to_json(path, orient="records", lines=True, double_precision=15)
        # Reload the data to ensure consistent values.
        df = pd.read_json(path, orient="records", lines=True)
        self.data = df.to_dict(orient="dict")
        logging.info(f"{self.name}| Seed = {SEED} | Training = {TRAINING} | Evaluating = {EVALUATING} | Generated "
                     f"data saved to '{path}'.")
        self.save_results()

    def get_data(self, lower: int = 0, upper: int = -1, training: bool = True, orientation: bool = False) -> list:
        """
        Get data to use for training or evaluation.
        :param lower: The starting joint.
        :param upper: The ending joint.
        :param training: If this is training data or evaluating data.
        :param orientation: If this data cares about the orientation or not.
        :return: The training or evaluation data as a list of dicts.
        """
        # Nothing to do if the robot is not valid.
        if not self.is_valid():
            logging.error(f"{self.name} | Get Data | Robot not configured.")
            return []
        # Nothing to do if this chain is not valid.
        lower, upper = self.validate_lower_upper(lower, upper)
        if self.chains[lower][upper] is None:
            logging.error(f"{self.name} | {lower + 1} to {upper + 1} | Get Data | Chain not valid.")
            return []
        # Get the portion of data requested.
        category = TRAINING_TITLE if training else EVALUATING_TITLE
        pose = TRANSFORM if orientation else POSITION
        data = self.data[lower][upper][category][pose]
        # If there is no data loaded for this, there is nothing to get.
        if data is None:
            logging.error(f"{self.name} | {lower + 1} to {upper + 1} | Get Data | No data.")
            return []
        # Get all instances in a list.
        values = []
        total = 0
        for title in data:
            amount = len(data[title])
            if amount > total:
                total = amount
        for i in range(total):
            value = {}
            for title in data:
                value[title] = data[title][i]
            values.append(value)
        return values

    def details(self, lower: int = 0, upper: int = -1) -> (str, int, int, int, bool, bool):
        """
        Get the details of a kinematic chain.
        :param lower: The starting joint.
        :param upper: The ending joint.
        :return: A formatted table of the chain, the number of revolute joints, the number of prismatic joints, the
        number of fixed links, if the chain has a dedicated TCP, and if there are limits.
        """
        # If not valid, there is nothing to display.
        if not self.is_valid():
            logging.error(f"{self.name} | Details | Robot not configured.")
            return "", 0, 0, 0, False, False
        # Get all values to perform.
        lower, upper = self.validate_lower_upper(lower, upper)
        chain = self.chains[lower][upper]
        if chain is None:
            logging.error(f"{self.name} | {lower + 1} to {upper + 1} | Details | Chain not valid.")
            return "", 0, 0, 0, False, False
        total = len(chain.links)
        # Define the headers which we will for sure have.
        headers = ["Link", "Position", "Orientation"]
        # Count the numbers of each type of link.
        revolute = 0
        prismatic = 0
        fixed = 0
        tcp = False
        limits = False
        for i in range(total):
            link = chain.links[i]
            # If this is the TCP, we are at the end so stop.
            if link.name == "TCP":
                tcp = True
                break
            # Determine the link type.
            if link.has_rotation:
                revolute += 1
            elif link.has_translation:
                prismatic += 1
            else:
                fixed += 1
            # Determine if this link has bounds.
            if link.bounds is not None and link.bounds != (-np.inf, np.inf):
                limits = True
        # If there are revolute joints, we need to display them.
        if revolute > 0:
            headers.append("Axis")
        # If there are prismatic joints, we need to display them.
        if prismatic > 0:
            headers.append("Translation")
        # If there are limits, we need to display them.
        if limits:
            headers.append("Limits")
        # Build the table data.
        data = []
        for i in range(total):
            link = chain.links[i]
            # We need the name which already has the joint type, position, and orientation.
            details = [link.name, neat(link.origin_translation), neat(link.origin_orientation)]
            # Display the rotational axis if this is a revolute joint.
            if revolute > 0:
                if link.rotation is None:
                    details.append("")
                else:
                    details.append(get_direction_details(link.rotation))
            # Display the translational axis if this is a prismatic joint.
            if prismatic > 0:
                if link.translation is None:
                    details.append("")
                else:
                    details.append(get_direction_details(link.translation))
            # Display limits if this has any.
            if limits:
                if link.joint_type == "fixed" or link.bounds is None or link.bounds == (-np.inf, np.inf):
                    details.append("")
                else:
                    details.append(neat(link.bounds))
            data.append(details)
        # Return all details.
        return tabulate(data, headers, tablefmt="presto"), revolute, prismatic, fixed, tcp, limits

    def prepare_llm(self, lower: int = 0, upper: int = -1, orientation: bool = False, additional: str = "") -> str:
        """
        Prepare information about the chain for a LLM.
        :param lower: The starting joint.
        :param upper: The ending joint.
        :param orientation: If orientation is being solved for.
        :param additional: Any additional instructions to be inserted.
        :return: The formatted prompt.
        """
        # If not valid, there is nothing to prepare for.
        if not self.is_valid():
            logging.error(f"{self.name} | Prepare LLM | Robot not configured.")
            return ""
        # Get all values to perform.
        lower, upper = self.validate_lower_upper(lower, upper)
        table, revolute, prismatic, fixed, has_tcp, limits = self.details(lower, upper)
        dof = revolute + fixed
        # Nothing to prepare to solve for if there are no degrees-of-freedom.
        if dof < 1:
            logging.error(f"{self.name} | {lower + 1} to {upper + 1} | Prepare LLM | No degrees of freedom.")
            return ""
        if dof < 2 and not has_tcp:
            logging.error(f"{self.name} | {lower + 1} to {upper + 1} | Prepare LLM | Only one link and no TCP.")
            return ""
        # If there is only a single degree-of-freedom, the position solution is the same as the whole transform.
        if dof < 2:
            orientation = False
        # Build the prompt.
        s = ("<INSTRUCTIONS>\nYou are tasked with producing a closed-form analytical solution for the inverse "
             f"kinematics of the {dof} degree{'s' if dof > 1 else ''}-of-freedom serial manipulator solving for the "
             f"position{' and orientation' if orientation else ''} of the {'TCP' if has_tcp else 'last link'} as "
             'detailed in the "DETAILS" section by completing the Python function provided in the "CODE" section. The '
             '"Position" and "Orientation" columns represent link coordinates in local space relative to their parent '
             'link. The positions are from the "xyx" attribute and the orientations are the "rpy" attribute from each '
             'link\'s "origin" element parsed from the URDF.')
        if revolute > 0:
            s += (' The "Axis" column in the table represents the rotational axis of the revolute '
                  f"link{'s' if revolute > 1 else ''}; return their values in radians.")
            if limits:
                s += f" and {'their' if revolute > 1 else 'the'} limits are in radians"
            s += "."
        if prismatic > 0:
            s += (' The "Translation" column in the table represents the movement axis of the prismatic '
                  f"link{'s' if prismatic > 1 else ''}.")
        if fixed > 0:
            s += (f" The fixed link{'s do' if fixed > 1 else ' does'} not have any movement; do not return anything "
                  f"for these links.")
        s += (" You are to respond with only the code for the completed inverse kinematics method with no additional "
              "text. Do not write any code to run the method for testing. You may use any methods included in Python, "
              "NumPy, SymPy, and SciPy to write your solution except for any interative optimization methods."
              f"{additional}\n</INSTRUCTIONS>\n<DETAILS>\n{table}\n</DETAILS>\n<CODE>\ndef inverse_kinematics(p: "
              "tuple[float, float, float]")
        if orientation:
            s += ", r: tuple[float, float, float]"
        reach = ' and orientation "r"' if orientation else ""
        if dof > 1:
            ret = "tuple[float"
            for i in range(1, dof):
                ret += ", float"
            ret += "]"
            ret_param = "A list of the values to set the links"
        else:
            ret = "float"
            ret_param = "The value to set the link"
        s += (f') -> {ret}:\n    """\n    Gets the joint values needed to reach position "p"{reach}.\n    :param p :The'
              f" position to reach in the form [x, y, z].")
        if orientation:
            s += "\n    :param r: The orientation to reach in radians in the form [x, y, z]."
        s += f'\n    :return: {ret_param} to for reaching position "p"{reach}.\n    """\n</CODE>'
        logging.info(f"{self.name} | {lower + 1} to {upper + 1} | Prompt prepared.")
        return s

    def forward_kinematics(self, lower: int = 0, upper: int = -1, joints: list[float] or None = None,
                           plot: bool = False, width: float = 10, height: float = 10,
                           target: list[float] or None = None) -> (list[list[float]], list[list[float]]):
        """
        Perform forward kinematics.
        :param lower: The starting joint.
        :param upper: The ending joint.
        :param joints: Values to set the joints to.
        :param plot: If the result should be plotted.
        :param width: The width to plot.
        :param height: The height to plot.
        :param target: The target to display in the plot.
        :return: The positions and orientations of all links from the forward kinematics.
        """
        # Ensure we can perform forward kinematics.
        if not self.is_valid():
            logging.error(f"{self.name} | Forward Kinematics | Robot not configured.")
            return [0, 0, 0], [0, 0, 0]
        # Get all values to perform.
        lower, upper = self.validate_lower_upper(lower, upper)
        chain = self.chains[lower][upper]
        if chain is None:
            logging.error(f"{self.name} | {lower + 1} to {upper + 1} | Forward Kinematics | Chain not valid.")
            return [0, 0, 0], [0, 0, 0]
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
        links = chain.forward_kinematics(values, True)
        # Get the positions and orientations of each link.
        positions = []
        orientations = []
        for forward in links:
            positions.append(list(forward[:3, 3]))
            orientations.append(list(Rotation.from_matrix(forward[:3, :3]).as_euler("xyz")))
        # Plot if we should.
        if plot:
            fig, ax = plot_utils.init_3d_figure()
            fig.set_size_inches(abs(width), abs(height))
            # If a target to display was passed, display it, otherwise display the end position.
            chain.plot(values, ax, positions[-1] if target is None else target)
            plt.title(chain.name)
            plt.tight_layout()
            plt.show()
        logging.debug(f"{self.name} | {lower + 1} to {upper + 1} | Forward kinematics | Joints = {controlled} | "
                      f"Position = {positions[-1]} | Orientation = {orientations[-1]}")
        return positions, orientations

    def inverse_kinematics(self, lower: int = 0, upper: int = -1, position: list[float] or None = None,
                           orientation: list[float] or None = None, plot: bool = False, width: float = 10,
                           height: float = 10) -> (list[float], float, float, float):
        """
        Perform inverse kinematics.
        :param lower: The starting joint.
        :param upper: The ending joint.
        :param position: The position to solve for.
        :param orientation: The orientation to solve for.
        :param plot: If the result should be plotted.
        :param width: The width to plot.
        :param height: The height to plot.
        :return: The solution joints, positional error, rotational error, and solving time.
        """
        # Ensure we can perform inverse kinematics.
        if not self.is_valid():
            logging.error(f"{self.name} | Inverse Kinematics | Robot not configured.")
            return [], np.inf, np.inf, np.inf
        # Set the joints to start at the midpoints.
        lower, upper = self.validate_lower_upper(lower, upper)
        if position is None:
            logging.warning(f"{self.name} | {lower + 1} to {upper + 1} | Inverse Kinematics | No target position was "
                            f"passed, solving for [0, 0, 0].")
            position = [0, 0, 0]
        # No point in solving for orientation when it is just one joint.
        if lower == upper:
            orientation = None
        chain = self.chains[lower][upper]
        if chain is None:
            logging.error(f"{self.name} | {lower + 1} to {upper + 1} | Inverse Kinematics | Chain not valid.")
            return [], np.inf, np.inf, np.inf
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
        start_time = time.perf_counter()
        values = chain.inverse_kinematics_frame(target, orientation_mode=None if orientation is None else "all",
                                                initial_position=values)
        elapsed = time.perf_counter() - start_time
        # Get the actual joint values, ignoring fixed links.
        solution = []
        for i in range(total):
            if chain.links[i].joint_type != "fixed":
                solution.append(values[i])
        # Get the reached position and orientations.
        true_positions, true_orientations = self.forward_kinematics(lower, upper, solution)
        true_position = true_positions[-1]
        true_orientation = true_orientations[-1]
        # Get the position error.
        distance = difference_distance(position, true_position)
        # Get the orientation error if it was being solved for.
        angle = 0 if orientation is None else difference_angle(orientation, true_orientation)
        # Plot if we should.
        if plot:
            fig, ax = plot_utils.init_3d_figure()
            fig.set_size_inches(abs(width), abs(height))
            # Show the goal position that should have been reached.
            chain.plot(values, ax, position)
            plt.title(chain.name)
            plt.tight_layout()
            plt.show()
        if orientation is not None:
            logging.debug(f"{self.name} | {lower + 1} to {upper + 1} | Inverse Kinematics | Target Position = "
                          f"{position} | Reached Position = {true_position} | Position Error = {distance} | Target "
                          f"Orientation = {orientation} | Reached Orientation = {true_orientation} | Orientation Error "
                          f"= {angle} | Solution = {solution} | Time = {elapsed} seconds")
        else:
            logging.debug(f"{self.name} | {lower + 1} to {upper + 1} | Inverse Kinematics | Target = {position} | "
                          f"Reached = {true_position} | Error = {distance} | Solution = {solution} | Time = {elapsed} "
                          "seconds")
        return solution, distance, angle, elapsed

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
        # If no upper value was passed, or it is more than there are joints, use the last joint.
        if upper < 0 or upper >= self.joints:
            upper = self.joints - 1
        # If no lower value was passed, use the first joint.
        if lower < 0:
            lower = 0
        # Ensure the lower value is at most equal to the upper value for a single joint chain.
        elif lower > upper:
            lower = upper
        # Return the updated lower and upper values.
        return lower, upper


class Solver:
    """
    Handle a solver attached to a robot.
    """

    def __init__(self, model: str, robot: Robot):
        """
        Load a solver.
        :param model: The name of the model.
        :param robot: The robot for the solver.
        """
        self.model = model
        self.robot = robot
        self.code = None
        # If the robot is invalid, there is nothing to do.
        if self.robot is None:
            logging.error(f"{self.model} | Robot is null.")
            self.interactions = os.path.join(INTERACTIONS, "_Invalid", self.model)
            self.solutions = os.path.join(SOLUTIONS, "_Invalid", self.model)
            self.results = os.path.join(RESULTS, "_Invalid", self.model)
            return
        # Cache folders.
        self.interactions = os.path.join(INTERACTIONS, self.robot.name, self.model)
        self.solutions = os.path.join(SOLUTIONS, self.robot.name, self.model)
        self.results = os.path.join(RESULTS, self.robot.name, self.model)
        # Ensure the robot is valid.
        if not robot.is_valid():
            logging.error(f"{self.model} | {self.robot.name} | Robot is not valid.")
            return
        # Load the code of all existing solvers.
        self.load_codes()
        logging.info(f"{self.model} | {self.robot.name} | Solver loaded.")
        self.save_prompts()

    def __str__(self) -> str:
        """
        Print as a string.
        :return: The name of this solver.
        """
        return self.model

    def save_prompts(self) -> None:
        """
        Save all valid initial prompts to text documents.
        :return:
        """
        # Nothing to load if the solver is not valid.
        if not self.is_valid():
            logging.error(f"{self.model} | Save Prompts | Solver is not valid.")
            return None
        # Loop all possible combinations.
        for lower in range(self.robot.joints):
            for upper in range(lower, self.robot.joints):
                for orientation in [False, True]:
                    # No solving for orientation with just one link.
                    if orientation and lower == upper:
                        break
                    # Try building the prompts for all modes.
                    for mode in [NORMAL, EXTEND, DYNAMIC]:
                        # Can only do the normal mode for single-link chains.
                        if mode != NORMAL and lower == upper:
                            break
                        # Get the prompt.
                        prompt = self.prepare_llm(lower, upper, orientation, mode, True)
                        # If no prompt is returned, there is nothing to do.
                        if prompt == "":
                            continue
                        # Save to a text file.
                        path = os.path.join(self.interactions,
                                            f"{lower}-{upper}-{TRANSFORM if orientation else POSITION}-{mode}")
                        os.makedirs(path, exist_ok=True)
                        with open(os.path.join(path, f"0-{MESSAGE}.txt"), "w") as file:
                            file.write(prompt)
        logging.info(f"{self.model} | Save Prompts | Valid initial prompts saved to '{self.interactions}'.")

    def load_codes(self) -> None:
        """
        Load all existing codes for the solver.
        :return: Nothing.
        """
        # Nothing to load if the solver is not valid.
        if not self.is_valid():
            logging.error(f"{self.model} | Load Codes | Solver is not valid.")
            return None
        # Load every possible solver.
        end = self.robot.joints
        for lower in range(end):
            for upper in range(lower, end):
                for orientation in [False, True]:
                    for mode in [NORMAL, EXTEND, DYNAMIC]:
                        # Suppress the error messages for solvers that do not exist.
                        self.load_code(lower, upper, orientation, mode, True)

    def load_code(self, lower: int = 0, upper: int = -1, orientation: bool = False, mode: str = NORMAL,
                  suppress: bool = False) -> None:
        """
        Load the code for a solver.
        :param lower: The starting joint.
        :param upper: The ending joint.
        :param orientation: If this data cares about the orientation or not.
        :param mode: The mode by which the code was achieved.
        :param suppress: If the error for the code not existing should be suppressed.
        :return: Nothing.
        """
        # Nothing to do if the solver is not valid.
        if not self.is_valid():
            logging.error(f"{self.model} | Load Code | Solver is not valid.")
            return None
        # Ensure valid values.
        lower, upper = self.robot.validate_lower_upper(lower, upper)
        if mode not in [NORMAL, EXTEND, DYNAMIC]:
            logging.warning(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Load Code | Mode "
                            f"'{mode}' not valid, using '{NORMAL}' instead.")
            mode = NORMAL
        # Get the name and path of what to load.
        solving = TRANSFORM if orientation else POSITION
        name = f"{lower}-{upper}-{solving}-{mode}"
        path = os.path.join(self.solutions, f"{name}.py")
        # Nothing to do if the file does not exist.
        if not os.path.exists(path):
            if not suppress:
                logging.error(f"{self.model} | {lower + 1} to {upper + 1} | Load Code | {solving} | {mode} | Solver "
                              f"'{path}' does not exist.")
            return None
        # Try to load the inverse kinematics method from the Python file.
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # If the method is not in the file, return.
            if not hasattr(module, "inverse_kinematics"):
                logging.error(f"{self.model} | {lower + 1} to {upper + 1} | Load Code | {solving} | {mode} | Solver "
                              f"'{path}' does not have the method 'inverse_kinematics'.")
                return None
            method = getattr(module, "inverse_kinematics")
        except Exception as e:
            logging.error(f"{self.model} | {lower + 1} to {upper + 1} | Load Code | {solving} | {mode} | Failed to load"
                          f" '{path}': {e}")
            return None
        # Cache the method.
        if self.code is None:
            self.code = {}
        if lower not in self.code:
            self.code[lower] = {}
        if upper not in self.code[lower]:
            self.code[lower][upper] = {}
        if solving not in self.code[lower][upper]:
            self.code[lower][upper][solving] = {}
        self.code[lower][upper][solving][mode] = method

    def run_code(self, lower: int = 0, upper: int = -1, mode: str = NORMAL, position: list[float] or None = None,
                 orientation: list[float] or None = None) -> (list[float] or None, float, str or None):
        """
        Run code for a chain
        :param lower: The starting joint.
        :param upper: The ending joint.
        :param mode: The mode by which the code was achieved.
        :param position: The position to solve for.
        :param orientation: The orientation to solve for.
        :return: The joints returned by the method, the time the method took, and an error message if there was one.
        """
        # Nothing to do if the solver is not valid.
        if not self.is_valid():
            logging.error(f"{self.model} | Run Code | Solver is not valid.")
            return None, 0, None
        # Nothing to do if there is no code loaded.
        if self.code is None:
            logging.error(f"{self.model} | Run Code | No codes loaded.")
            return None, 0, None
        # Ensure valid values.
        lower, upper = self.robot.validate_lower_upper(lower, upper)
        if lower not in self.code:
            logging.error(f"{self.model} | {lower + 1} to {upper + 1} | Run Code | No codes loaded for starting at "
                          f"{lower + 1}.")
            return None, 0, None
        if upper not in self.code[lower]:
            logging.error(f"{self.model} | {lower + 1} to {upper + 1} | Run Code | No codes loaded for starting at "
                          f"{lower + 1} and ending at {upper + 1}.")
            return None, 0, None
        solving = POSITION if orientation is None else TRANSFORM
        if solving not in self.code[lower][upper]:
            logging.error(f"{self.model} | {lower + 1} to {upper + 1} | Run Code | No codes loaded for starting at "
                          f"{lower + 1}, ending at {upper + 1}, and solving for '{solving}'.")
            return None, 0, None
        if mode not in [NORMAL, EXTEND, DYNAMIC]:
            logging.warning(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Run Code | Mode "
                            f"'{mode}' not valid, using '{NORMAL}' instead.")
            mode = NORMAL
        if mode not in self.code[lower][upper][solving]:
            logging.error(f"{self.model} | {lower + 1} to {upper + 1} | Run Code | No codes loaded for starting at "
                          f"{lower + 1}, ending at {upper + 1}, solving for '{solving}', and generated using '{mode}' "
                          "mode.")
            return None, 0, None
        # Ensure a position.
        if position is None:
            logging.warning(f"{self.model} | {lower + 1} to {upper + 1} | Run Code | {solving} | {mode} | No position "
                            "passed; solving for [0, 0, 0].")
            position = [0, 0, 0]
        position = tuple(position)
        joints = []
        message = None
        # Run the code passing the orientation if we should.
        if orientation is None:
            start_time = time.perf_counter()
            try:
                joints = self.code[lower][upper][solving][mode](position)
                elapsed = time.perf_counter() - start_time
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                message = traceback.format_exc()
                logging.error(f"{self.model} | {lower + 1} to {upper + 1} | Run Code | {solving} | {mode} | Error: {e}")
        else:
            orientation = tuple(orientation)
            start_time = time.perf_counter()
            try:
                joints = self.code[lower][upper][solving][mode](position, orientation)
                elapsed = time.perf_counter() - start_time
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                message = traceback.format_exc()
                logging.error(f"{self.model} | {lower + 1} to {upper + 1} | Run Code | {solving} | {mode} | Error: {e}")
        # Parse the joints.
        if joints is not None:
            try:
                temp = []
                for joint in joints:
                    temp.append(joint)
                joints = temp
            except Exception as e:
                logging.error(f"{self.model} | {lower + 1} to {upper + 1} | Run Code | {solving} | {mode} | Joints "
                              f"could not be cast to a list: {e}")
                joints = None
        else:
            logging.error(f"{self.model} | {lower + 1} to {upper + 1} | Run Code | {solving} | {mode} | No joints "
                          f"returned.")
        return joints, elapsed, message

    def prepare_feedback(self, lower: int = 0, upper: int = -1, orientation: bool = False,
                         mode: str = NORMAL) -> str:
        """
        Prepare a feedback prompt for the LLM.
        :param lower: The starting joint.
        :param upper: The ending joint.
        :param orientation: If we want to solve for orientation.
        :param mode: The solving mode to use.
        :return: The feedback prompt for the LLM.
        """
        # Nothing to do if the solver is not valid.
        if not self.is_valid():
            logging.error(f"{self.model} | Prepare Feedback | Solver is not valid.")
            return ""
        # Ensure valid values.
        lower, upper = self.robot.validate_lower_upper(lower, upper)
        solving = POSITION if orientation is None else TRANSFORM
        if mode not in [NORMAL, EXTEND, DYNAMIC]:
            logging.warning(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare Feedback | Mode "
                            f"'{mode}' not valid, using '{NORMAL}' instead.")
            mode = NORMAL
        # Get the data to run the code against.
        data = self.robot.get_data(lower, upper, True, orientation)
        # If there is no data, there is nothing to give feedback on.
        if len(data) < 1:
            logging.error(f"{self.model} | {lower + 1} to {upper + 1} | {solving} | {mode} | No data.")
            return ""
        # Store what to respond with.
        errors = []
        failures = []
        number = upper - lower + 1
        for point in data:
            # Determine what to test against.
            target_position = point["Position"]
            target_orientation = point["Orientation"] if orientation else None
            # Run the code.
            joints, elapsed, error = self.run_code(lower, upper, mode, target_position, target_orientation)
            # If we got an error, save it.
            if error is not None:
                if error not in errors:
                    errors.append(error)
                if len(errors) >= EXAMPLES:
                    break
                continue
            # See if we got a valid number of joints back.
            if joints is None:
                error = f"Returned no joints - expected {number}."
                if error not in errors:
                    errors.append(error)
                if len(errors) >= EXAMPLES:
                    break
                continue
            got = len(joints)
            if got != number:
                error = f"Returned the wrong number of joints - expected {number} but got {got}."
                if error not in errors:
                    errors.append(error)
                if len(errors) >= EXAMPLES:
                    break
                continue
            # If there are any errors, that is all we will give feedback about, so don't test anything else.
            if len(errors) > 0:
                continue
            # See if we reached the target.
            positions, orientations = self.robot.forward_kinematics(lower, upper, joints)
            distance = difference_distance(target_position, positions[-1])
            angle = 0 if orientation is None else difference_angle(target_orientation, orientations[-1])
            # If we did, this was a success so continue.
            if reached(distance, angle):
                continue
            # Otherwise, detail what the issue was.
            a = f" and orientation {neat(target_orientation)}" if orientation else ""
            b = f" and orientation {neat(orientations[-1])}" if orientation else ""
            failures.append(f"Failed to reach position {neat(target_position)}{a}. Instead reached position "
                            f"{neat(positions[-1])}{b}. The correct joint values were {neat(point['Joints'])} and the "
                            f"joints produced by the code were {neat(joints)}.")
        total = len(errors)
        if total > 0:
            plural = "s" if total > 1 else ""
            s = (f"<FEEDBACK>\nThe code was tested on multiple trials with valid inputs but encountered the following "
                 f"error{plural}:")
            for error in errors:
                s += f"\n{error}"
            return f"{s}\n</FEEDBACK>"
        total = min(EXAMPLES, len(failures))
        if total < 1:
            return ""
        s = (f"<FEEDBACK>\nThe code was tested on multiple trials with valid inputs but failed to reach all targets. "
             f"The solution for the joints generated from a working inverse kinematics solver have been provided for "
             f"you to learn from:")
        for i in range(total):
            s += f"\n{failures[i]}"
        return f"{s}\n</FEEDBACK>"

    def prepare_llm(self, lower: int = 0, upper: int = -1, orientation: bool = False, mode: str = NORMAL,
                    suppress: bool = False) -> str:
        """
        Prepare an initial prompt for the LLM.
        :param lower: The starting joint.
        :param upper: The ending joint.
        :param orientation: If we want to solve for orientation.
        :param mode: The solving mode to use.
        :param suppress: If the error for the code not existing should be suppressed.
        :return: The initial prompt for the LLM.
        """
        # Nothing to do if the solver is not valid.
        if not self.is_valid():
            logging.error(f"{self.model} | Prepare LLM | Solver is not valid.")
            return ""
        # Ensure valid values.
        lower, upper = self.robot.validate_lower_upper(lower, upper)
        if mode not in [NORMAL, EXTEND, DYNAMIC]:
            logging.warning(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM  | Mode "
                            f"'{mode}' not valid, using '{NORMAL}' instead.")
            mode = NORMAL
        # Cannot do orientation if just a single joint.
        if lower == upper:
            orientation = False
        # Can only do normal mode for single joint chains.
        if mode == NORMAL or lower == upper:
            prompt = self.robot.prepare_llm(lower, upper, orientation)
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | Normal prompt "
                         f"prepared.")
            return prompt
        # Cache what is being solved for file outputs.
        solving = TRANSFORM if orientation else POSITION
        # Extending prompting mode.
        if mode == EXTEND:
            # We need to load the results of the lower portion, which when just one joint is the normal mode.
            previous = upper - 1
            previous_mode = NORMAL if lower == previous else EXTEND
            path = os.path.join(self.solutions,
                                f"{lower}-{previous}-{solving}-{previous_mode}.py")
            # Cannot prepare a prompt in this mode if the chain to extend does not exist.
            if not os.path.exists(path):
                if not suppress:
                    logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | "
                                  f"Cannot load an extending prompt as '{path}' does not exist.")
                return ""
            existing_feedback = self.prepare_feedback(lower, previous, orientation, previous_mode)
            # Only perform an extending prompt if the previous chain was successful.
            if existing_feedback != "":
                logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Not performing an "
                             f"extending prompt as '{path}' is not perfectly successful.")
                return ""
            # Add the extending prompt portions.
            total = upper - lower
            plural = "s" if total > 1 else ""
            additional = (f" To help you, a solution for solving the sub-chain of the first {total} link{plural} is "
                          f'provided in the "EXISTING" section. This code solved the sub-chain assuming link '
                          f"{total + 1} was the position{' and orientation' if orientation else ''} being solved for. "
                          f"You can use this solution as a starting point to extend for the entire chain.")
            prompt = self.robot.prepare_llm(lower, upper, orientation, additional)
            prompt += "\n<EXISTING>\n"
            with open(path, "r") as file:
                prompt += file.read().strip()
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Extended prompt prepared.")
            return f"{prompt}\n</EXISTING>"
        # TODO - Implement dynamic mode prompt building.
        logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | Dynamic prompts "
                      f"not yet implemented.")
        return ""

    def is_valid(self) -> bool:
        """
        Ensure the solver is valid.
        :return: True if the robot is valid, false otherwise.
        """
        return self.robot is not None and self.robot.is_valid()


def get_direction_details(vector) -> str:
    """
    Get a string containing the direction details for a vector.
    :param vector:
    :return: The string containing the direction details for a vector.
    """
    # Split the vector into its components.
    x, y, z = vector
    # If the vector is not aligned, simply get the cleaned version of the vector.
    aligned = is_aligned(x) and is_aligned(y) and is_aligned(z)
    if not aligned:
        return neat(vector)
    # Determine the number of axes which are active.
    active = 0
    if x != 0:
        active += 1
    if y != 0:
        active += 1
    if z != 0:
        active += 1
    # If none are active, return nothing.
    if active < 1:
        return ""
    # If one is active, return only it.
    elif active == 1:
        if x != 0:
            return get_direction_value(x, "X")
        if y != 0:
            return get_direction_value(y, "Y")
        return get_direction_value(z, "Z")
    # Otherwise, build and return all active axes.
    s = "["
    if x != 0:
        t = get_direction_value(x, "X")
        if s == "[":
            s += t
        else:
            s += f", {t}"
    if y != 0:
        t = get_direction_value(y, "Y")
        if s == "[":
            s += t
        else:
            s += f", {t}"
    if z != 0:
        t = get_direction_value(z, "Z")
        if s == "[":
            s += t
        else:
            s += f", {t}"
    return f"{s}]"


def is_aligned(value) -> bool:
    """
    Determine if an axis component is perfectly aligned being a zero or one.
    :param value: The axis component.
    :return: True if the axis component is perfectly aligned being a zero or one, false otherwise.
    """
    return value == 0 or value == 1


def get_direction_value(value, representation: str) -> str:
    """
    Get a clean direction value.
    :param value: The axis component value.
    :param representation: The component this represents.
    :return: A clean direction value for the axis component.
    """
    return representation if value > 0 else f"-{representation}"


def neat(value: float or list or tuple or np.array) -> str:
    """
    Format a float value with no trailing zeros.
    :param value: The float value.
    :return: The value as a formatted string.
    """
    # If this contains multiple elements, clean every one.
    if isinstance(value, (list, tuple, np.ndarray)):
        count = len(value)
        s = "["
        for i in range(count):
            if i == 0:
                s += neat(value[i])
            else:
                s += f", {neat(value[i])}"
        return f"{s}]"
    # Otherwise, clean the value.
    value = str(value).rstrip('0').rstrip('.')
    return "0" if value == "" else value


def reached(distance: float = 0, angle: float = 0) -> bool:
    """
    Determine if a target has been reached.
    :param distance: The distance.
    :param angle: The angle.
    :return: True if it has been reached, false otherwise.
    """
    return distance <= DISTANCE_ERROR and angle <= ANGLE_ERROR


def difference_distance(a, b) -> float:
    """
    Get the difference between two positions.
    :param a: First position.
    :param b: Second position.
    :return: The differences between the two positions.
    """
    return np.sqrt(sum([(x - y) ** 2 for x, y in zip(a, b)]))


def difference_angle(a: float or int = 0, b: float or int = 0) -> float:
    """
    Get the difference between two angles.
    :param a: First angle.
    :param b: Second angle.
    :return: The differences between the two angle.
    """
    return sum([abs(x - y) for x, y in zip(a, b)])


def get_files(directory) -> list[str]:
    """
    Get all files in a directory.
    :param directory: The directory to get the files from.
    :return: The names of all files in the directory.
    """
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def llm_ik(robots: str or list[str] or None = None, models: str or list[str] or None = None,
           orientation: bool or None = False, types: str or list[str] or None = None, feedbacks: int = FEEDBACKS,
           examples: int = EXAMPLES, training: int = TRAINING, evaluating: int = EVALUATING, seed: int = SEED,
           distance_error: float = DISTANCE_ERROR, angle_error: float = ANGLE_ERROR, length: int or None = None,
           run: bool = False, cwd: str or None = None, level: str = "INFO") -> None:
    """
    Run LLM inverse kinematics.
    :param robots: The names of the robots.
    :param models: The names of the LLMs.
    :param orientation: If we want to solve for position, transform, or both being none.
    :param types: The solving types.
    :param feedbacks: The max number of times to give feedback.
    :param examples: The number of examples to give with feedbacks.
    :param training: The number of training samples.
    :param evaluating: The number of evaluating samples.
    :param seed: The samples generation seed.
    :param distance_error: The acceptable distance error.
    :param angle_error: The acceptable angle error.
    :param length: The maximum chain length to solve.
    :param run: Enable API running.
    :param cwd: The working directory.
    :param level: The logging level.
    :return: Nothing.
    """
    # Set the logging level.
    level = level.upper()
    if level == "CRITICAL" or level == "FATAL":
        level = logging.CRITICAL
    elif level == "ERROR":
        level = logging.ERROR
    elif level == "WARNING" or level == "WARN":
        level = logging.WARNING
    elif level == "INFO":
        level = logging.INFO
    elif level == "DEBUG":
        level = logging.DEBUG
    else:
        level = logging.NOTSET
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")
    # If no directory was passed, run in the current working directory.
    if cwd is None:
        cwd = os.getcwd()
        logging.info(f"Using '{cwd}' as the working directory.")
    # Otherwise, check if the passed directory exists.
    elif not os.path.exists(cwd):
        logging.error(f"Working directory of '{cwd}' does not exist.")
        return None
    logging.info(f"Set '{cwd}' as the working directory.")
    # Set all paths relative to the working directory and make sure they exist.
    global ROBOTS
    global MODELS
    global PROVIDERS
    global KEYS
    global INFO
    global INTERACTIONS
    global SOLUTIONS
    global RESULTS
    ROBOTS = os.path.join(cwd, ROBOTS)
    os.makedirs(ROBOTS, exist_ok=True)
    MODELS = os.path.join(cwd, MODELS)
    os.makedirs(MODELS, exist_ok=True)
    PROVIDERS = os.path.join(cwd, PROVIDERS)
    os.makedirs(PROVIDERS, exist_ok=True)
    KEYS = os.path.join(cwd, KEYS)
    os.makedirs(KEYS, exist_ok=True)
    INFO = os.path.join(cwd, INFO)
    os.makedirs(INFO, exist_ok=True)
    INTERACTIONS = os.path.join(cwd, INTERACTIONS)
    os.makedirs(INTERACTIONS, exist_ok=True)
    SOLUTIONS = os.path.join(cwd, SOLUTIONS)
    os.makedirs(SOLUTIONS, exist_ok=True)
    RESULTS = os.path.join(cwd, RESULTS)
    os.makedirs(RESULTS, exist_ok=True)

    # Ensure the passed types are valid.
    acceptable = [NORMAL, EXTEND, DYNAMIC]
    # If noe were passed, use all.
    if types is None:
        types = acceptable
    # If one was passed, but it is not a valid option, there is nothing to do.
    elif isinstance(types, str):
        if types not in acceptable:
            logging.error(f"Solving type of '{types}' is not an option; nothing to perform.")
            return None
        types = [types]
    # Ensure all list options are valid.
    else:
        found = []
        for t in types:
            if t not in acceptable:
                logging.warning(f"Solving type of '{t}' is not an option; removing it.")
            found.append(t)
        if len(found) < 1:
            logging.error("No valid solving types; nothing to perform.")
            return None
        # Ensure they are in the order of normal, extend, and then dynamic.
        types = sorted(found, reverse=True)
    logging.info(f"Solving in the following modes: {types}.")
    # Get the orientation types we wish to solve for.
    if orientation is None:
        logging.info("Solving for both position and transform.")
        orientation = [False, True]
    else:
        logging.info(f"Solving for only {'transform' if orientation else 'position'}.")
        orientation = [orientation]
    # Ensure all other values are valid and assigned.
    if feedbacks < 0:
        feedbacks = 0
    if training < 1:
        logging.warning("Must have at least one training sample.")
        training = 1
    global TRAINING
    TRAINING = training
    logging.info(f"Training with {TRAINING} sample{'' if TRAINING == 1 else 's'}.")
    if evaluating < 1:
        logging.warning("Must have at least one evaluating sample.")
        evaluating = 1
    global EVALUATING
    EVALUATING = evaluating
    logging.info(f"Evaluating with {EVALUATING} sample{'' if EVALUATING == 1 else 's'}.")
    global FEEDBACKS
    FEEDBACKS = feedbacks
    logging.info(f"Providing {FEEDBACKS} feedback{'' if FEEDBACKS == 1 else 's'}.")
    if examples < 1:
        logging.warning("Examples must be at minimum one.")
        examples = 1
    elif examples > TRAINING:
        logging.warning(f"Examples must be at most the training size of {TRAINING}.")
        examples = TRAINING
    global EXAMPLES
    EXAMPLES = examples
    logging.info(f"Giving feedbacks with {EXAMPLES} example{'' if EXAMPLES == 1 else 's'}.")
    global SEED
    SEED = seed
    logging.info(f"Using the seed {SEED}.")
    if distance_error < 0:
        logging.warning("Distance error must be at least zero.")
        distance_error = 0
    global DISTANCE_ERROR
    DISTANCE_ERROR = distance_error
    logging.info(f"Acceptable distance error is {distance_error}.")
    if angle_error < 0:
        logging.warning("Angle error must be at least zero.")
        angle_error = 0
    global ANGLE_ERROR
    ANGLE_ERROR = angle_error
    logging.info(f"Acceptable angle error is {angle_error}°.")
    if length is None:
        logging.info(f"Solving chains of all lengths.")
    else:
        if length < 1:
            logging.info("Solving chain lengths must be at least one.")
            length = 1
        logging.info(f"Solving chains up to a length of {length}.")
    global RUN
    RUN = run
    logging.info("Running LLM API calls." if RUN else "Not running LLM API calls.")
    # If there are no robots, there is nothing to do.
    existing = get_files(ROBOTS)
    if len(existing) < 1:
        logging.error(f"No robots in '{ROBOTS}'.")
        return None
    # If no robots were passed, run all of them.
    if robots is None:
        robots = existing
    # Otherwise, if a string was passed, load it if it exists.
    elif isinstance(robots, str):
        if robots not in existing and f"{robots}.urdf" not in existing:
            logging.error(f"Robot '{robots}' does not exist in '{ROBOTS}'.")
            return None
        robots = [robots]
    # Otherwise, if a list, ensure all passed robots exist.
    else:
        found = []
        for robot in robots:
            if robot not in existing and f"{robot}.urdf" not in existing:
                logging.warning(f"Robot '{robot}' does not exist in '{ROBOTS}'; removing it.")
                continue
            found.append(robot)
        if len(found) < 1:
            logging.error(f"No valid robots were passed; nothing to perform on.")
            return None
        robots = found
    # Load all robots.
    created = []
    for name in robots:
        robot = Robot(name)
        if robot.is_valid():
            created.append(robot)
    if len(created) < 1:
        logging.error("No robots could be successfully loaded; nothing to perform on.")
        return None
    robots = created
    total = len(robots)
    logging.info(f"{total} robot{'s' if total > 1 else ''} loaded.")
    # See which models exist in the core folder.
    found = get_files(MODELS)
    # Clean out only to the models we are interested in.
    total = len(found)
    # If there are no models in the first place, there is nothing to check.
    if total < 1:
        logging.warning("No models; can only perform built-in IKPy inverse kinematics.")
        models = []
    # Otherwise, ensure our passed models are valid.
    else:
        # Clean the names of models.
        for i in range(total):
            found[i] = found[i].replace(".txt", "")
        # If no models were passed, use all found in the folder.
        if models is None:
            logging.info(f"Loading{' all' if total > 1 else ''} {total} model{'s' if total > 1 else ''}.")
            models = found
        # If it was a string, make sure it exists.
        elif isinstance(models, str):
            if models not in found:
                logging.error(f"Model '{models}' does not exist in '{MODELS}'; can only perform built-in IKPy inverse "
                              f"kinematics.")
                models = []
                total = 0
            else:
                logging.info(f"Loading '{models}'.")
                models = [models]
                total = 1
        # If it was a list, make sure they all are created.
        else:
            selected = []
            for model in models:
                if model not in found:
                    logging.error(f"Model '{model}' does not exist in '{MODELS}'; removing it.")
                else:
                    selected.append(model)
            models = selected
            total = len(models)
            if total < 1:
                logging.error("No models being loaded; can only perform built-in IKPy inverse kinematics.")
            else:
                logging.info(f"Loading {total} model{'s' if total > 1 else ''}.")
    # If there is at least one model we should load, let us try to fully load it.
    if total > 0:
        created = []
        # Try for every LLM paired with every robot.
        for name in models:
            for robot in robots:
                model = Solver(name, robot)
                if model.is_valid():
                    created.append(model)
        models = created
        total = len(models)
        if total < 1:
            logging.warning("No models loaded; can only perform built-in IKPy inverse kinematics.")
        else:
            logging.info(f"Loaded {total} model{'s' if total > 1 else ''}.")
    # TODO - Actually run stuff.


if __name__ == "__main__":
    # Configure the argument parser.
    parser = argparse.ArgumentParser(description="LLM Inverse Kinematics")
    parser.add_argument("-r", "--robots", type=str or list[str] or None, default=None, help="The names of the robots.")
    parser.add_argument("-m", "--models", type=str or list[str] or None, default=None, help="The names of the LLMs.")
    parser.add_argument("-o", "--orientation", type=bool or None, default=False, help="If we want to solve for "
                                                                                      "position, transform, or both "
                                                                                      "being none.")
    parser.add_argument("-t", "--types", type=str or list[str] or None, default=None, help="The solving types.")
    parser.add_argument("-f", "--feedbacks", type=int, default=FEEDBACKS, help="The max number of times to give "
                                                                               "feedback.")
    parser.add_argument("-e", "--examples", type=int, default=EXAMPLES, help="The number of examples to give with "
                                                                             "feedbacks.")
    parser.add_argument("-a", "--training", type=int, default=TRAINING, help="The number of training samples.")
    parser.add_argument("-v", "--evaluating", type=int, default=EVALUATING, help="The number of evaluating samples.")
    parser.add_argument("-s", "--seed", type=int, default=SEED, help="The samples generation seed.")
    parser.add_argument("-d", "--distance", type=float, default=DISTANCE_ERROR, help="The acceptable distance error.")
    parser.add_argument("-n", "--angle", type=float, default=ANGLE_ERROR, help="The acceptable angle error.")
    parser.add_argument("-l", "--length", type=int or None, default=None, help="The maximum chain length to solve.")
    parser.add_argument("-c", "--cwd", type=str or None, default=None, help="The working directory.")
    parser.add_argument("-u", "--run", action="store_true", help="Enable API running.")
    parser.add_argument("-g", "--logging", type=str, default="INFO", help="The logging level.")
    args = parser.parse_args()
    # Run the program.
    llm_ik(args.robots, args.models, args.orientation, args.types, args.feedbacks, args.examples, args.training,
           args.evaluating, args.seed, args.distance, args.angle, args.length, args.run, args.cwd, args.logging)
