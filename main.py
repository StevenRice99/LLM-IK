import copy
import importlib
import importlib.util
import logging
import os.path
import random
import time
import warnings

import ikpy.chain
import ikpy.utils.plot as plot_utils
import numpy as np
import pandas as pd
from ikpy.link import URDFLink
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from tabulate import tabulate

MODELS = "Models"
INFO = "Info"
INTERACTIONS = "Interactions"
SOLUTIONS = "Solutions"
RESULTS = "Results"

NORMAL = "Normal"
EXTEND = "Extend"
DYNAMIC = "Dynamic"

MESSAGE = "Message"
RESPONSE = "Response"

POSITION = "Position"
TRANSFORM = "Transform"
TRAINING_TITLE = "Training"
EVALUATING_TITLE = "Evaluating"

TRAINING = 1
EVALUATING = 1
SEED = 42

BOUND = 2 * np.pi

# Set up logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


class Robot:
    """
    Handle all aspects of serial robots.
    """

    def __init__(self, name: str, models: str = MODELS, info: str = INFO, training: int = TRAINING,
                 evaluating: int = EVALUATING, seed: int = SEED):
        """
        Initialize the robot.
        :param name: The name of the URDF to load.
        :param models: The folder models are stored in.
        :param info: The folder to save info about chains to.
        :param training: The number of poses to use for training the LLM.
        :param evaluating: The number of poses to use for evaluating the LLMs.
        :param seed: The seed to use for generating poses.
        """
        self.name = os.path.splitext(name)[0]
        # Cache the to save info to.
        self.info = os.path.join(os.getcwd(), info, self.name)
        self.chains = None
        self.joints = 0
        self.training = 0
        self.evaluating = 0
        self.data = {}
        # Nothing to do if the file does not exist.
        path = os.path.join(os.getcwd(), models, name)
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
        # Set the seed for generating training and evaluating instances.
        self.load_data(training, evaluating, seed)
        logging.info(f"{self.name} | Loaded | {self.joints} Joints | Info saved to '{self.info}'.")

    def __str__(self) -> str:
        """
        Get the table of the details of the full robot.
        :return: The table of the details of the full robot.
        """
        return self.details()[0]

    def load_data(self, training: int = TRAINING, evaluating: int = EVALUATING, seed: int = SEED) -> None:
        """
        Load data for the robot to use.
        :param training: The number of training data instances.
        :param evaluating: The number of evaluation data instances.
        :param seed: The seed to generate the data with.
        :return: Nothing.
        """
        # Clear any previous data.
        self.data = {}
        # Set the random seed.
        random.seed(seed)
        np.random.seed(seed)
        # Ensure valid data amounts.
        if training < 1:
            training = 1
        if evaluating < 1:
            evaluating = 1
        self.training = training
        self.evaluating = evaluating
        # If there is already data for this configuration, load it.
        path = os.path.join(self.info, f"{seed}-{training}-{evaluating}.json")
        if os.path.exists(path):
            df = pd.read_json(path, orient="records", lines=True)
            self.data = df.to_dict(orient="dict")
            logging.info(f"{self.name}| Seed = {seed} | Training = {training} | Evaluating = {evaluating} | Generated "
                         f"data loaded from '{path}'.")
            return
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
                    instances = training if part == TRAINING_TITLE else evaluating
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
                logging.info(f"{self.name} | {lower + 1} to {upper + 1} | Seed = {seed} | Training = {training} | "
                             f"Evaluating = {evaluating} | Data generated.")
        # Save the newly generated data.
        os.makedirs(self.info, exist_ok=True)
        df = pd.DataFrame(self.data)
        df.to_json(path, orient="records", lines=True, double_precision=15)
        # Reload the data to ensure consistent values.
        df = pd.read_json(path, orient="records", lines=True)
        self.data = df.to_dict(orient="dict")
        logging.info(f"{self.name}| Seed = {seed} | Training = {training} | Evaluating = {evaluating} | Generated "
                     f"data saved to '{path}'.")

    def get_data(self, lower: int = 0, upper: int = -1, training: bool = True, orientation: bool = False,
                 start: int = 0, count: int = 1) -> list:
        """
        Get data to use for training or evaluation.
        :param lower: The starting joint.
        :param upper: The ending joint.
        :param training: If this is training data or evaluating data.
        :param orientation: If this data cares about the orientation or not.
        :param start: The starting index of data to request.
        :param count: The number of data entries to request.
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
        # Ensure valid values.
        if start < 0:
            start = 0
        if count < 1:
            count = 1
        end = start + count
        # If there are fewer data points than the range requested, clamp it.
        for title in data:
            instances = len(data[title])
            if instances < end:
                end = instances
                logging.warning(f"{self.name} | {lower + 1} to {upper + 1} | Get Data | Cannot get {count} instances "
                                f"from index {start} as there are {end} instances total; clamping.")
                break
        if start >= end:
            start = end - 1
            logging.warning(f"{self.name} | {lower + 1} to {upper + 1} | Get Data | Start was more than {end}; clamped "
                            f"to {start}.")
        # Get all instances in a list.
        values = []
        for i in range(start, end):
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
        end_time = time.perf_counter()
        elapsed = end_time - start_time
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

    def __init__(self, model: str, robot: Robot, interactions: str = INTERACTIONS, solutions: str = SOLUTIONS,
                 results: str = RESULTS):
        """
        Load a solver.
        :param model: The name of the model.
        :param robot: The robot for the solver.
        :param interactions: The folder to save and load interactions to and from.
        :param solutions: The folder to save and load solutions to and from.
        :param results: The folder to save results to.
        """
        self.model = model
        self.robot = robot
        self.code = None
        # If the robot is invalid, there is nothing to do.
        if self.robot is None:
            logging.error(f"{self.model} | Robot is null.")
            self.interactions = os.path.join(os.getcwd(), interactions, "_Invalid", self.model)
            self.solutions = os.path.join(os.getcwd(), solutions, "_Invalid", self.model)
            self.results = os.path.join(os.getcwd(), results, "_Invalid", self.model)
            return
        # Cache folders.
        self.interactions = os.path.join(os.getcwd(), interactions, self.robot.name, self.model)
        self.solutions = os.path.join(os.getcwd(), solutions, self.robot.name, self.model)
        self.results = os.path.join(os.getcwd(), results, self.robot.name, self.model)
        # Ensure the robot is valid.
        if not robot.is_valid():
            logging.error(f"{self.model} | {self.robot.name} | Robot is not valid.")
            return
        # Load the code of all existing solvers.
        self.load_codes()
        logging.info(f"{self.model} | {self.robot.name} | Solver loaded.")

    def __str__(self) -> str:
        """
        Print as a string.
        :return: The name of this solver.
        """
        return self.model

    def load_codes(self) -> None:
        """
        Load all existing codes for the solver.
        :return: Nothing.
        """
        # Nothing to load if the solver is not valid.
        if not self.is_valid():
            logging.error(f"{self.model} | Load Codes | Solver is not valid.")
            return
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
            return
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
            return
        # Try to load the inverse kinematics method from the Python file.
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # If the method is not in the file, return.
            if not hasattr(module, "inverse_kinematics"):
                logging.error(f"{self.model} | {lower + 1} to {upper + 1} | Load Code | {solving} | {mode} | Solver "
                              f"'{path}' does not have the method 'inverse_kinematics'.")
                return
            method = getattr(module, "inverse_kinematics")
        except Exception as e:
            logging.error(f"{self.model} | {lower + 1} to {upper + 1} | Load Code | {solving} | {mode} | Failed to load"
                          f" '{path}': {e}")
            return
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

    def prepare_llm(self, lower: int = 0, upper: int = -1, orientation: bool = False, mode: str = NORMAL) -> str:
        """
        Prepare an initial prompt for the LLM.
        :param lower: The starting joint.
        :param upper: The ending joint.
        :param orientation: If we want to solve for orientation.
        :param mode: The solving mode to use.
        :return:
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
                                f"{lower}-{previous}-{previous_mode}-{solving}.py")
            # Cannot prepare a prompt in this mode if the chain to extend does not exist.
            if not os.path.exists(path):
                logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | Cannot "
                              f"load an extending prompt as '{path}' does not exist.")
                return ""
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
        :return: True if the sovler's robot is valid, false otherwise.
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


def reached(distance: float = 0, angle: float = 0, distance_error: float = 0.001, angle_error: float = 0.001) -> bool:
    """
    Check if a robot has reached a target
    :param distance: The distance from the target.
    :param angle: The angle from the target.
    :param distance_error: The maximum acceptable positional error.
    :param angle_error: The maximum acceptable orientation error.
    :return: True if the target was reached, false otherwise.
    """
    return distance <= distance_error and angle <= angle_error


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


def main() -> None:
    """
    Handle main program operations.
    :return: Nothing.
    """
    ur5 = Robot("UR5.urdf")
    #solver = Solver("gpt-4o", ur5)
    ur5.get_data(training=True)


if __name__ == "__main__":
    main()
