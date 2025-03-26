import argparse
import ast
import copy
import importlib
import importlib.util
import json
import logging
import math
import numbers
import os.path
import random
import re
import time
import traceback
import warnings
from decimal import Decimal
from typing import Any

import ikpy.chain
import ikpy.utils.plot as plot_utils
import numpy as np
import pandas as pd
from func_timeout import func_timeout, FunctionTimedOut
from ikpy.link import URDFLink
from matplotlib import pyplot as plt
from num2words import num2words
from openai import NOT_GIVEN, OpenAI
from scipy.spatial.transform import Rotation
from tabulate import tabulate

# Folders.
ROBOTS = "Robots"
MODELS = "Models"
PROVIDERS = "Providers"
KEYS = "Keys"
INFO = "Info"
INTERACTIONS = "Interactions"
ELAPSED = "Elapsed"
SOLUTIONS = "Solutions"
RESULTS = "Results"
TOKENS = "Tokens"

# Execution modes.
NORMAL = "Normal"
EXTEND = "Extend"
DYNAMIC = "Dynamic"
CUMULATIVE = "Cumulative"
TRANSFER = "Transfer"

# API interaction file naming.
MESSAGE_PROMPT = "Prompt"
MESSAGE_FEEDBACK = "Feedback"
MESSAGE_FORWARD = "Forward"
MESSAGE_TEST = "Test"
MESSAGE_DONE = "Done"
RESPONSE = "Response"
INHERITED = "Inherited"

# Data naming.
POSITION = "Position"
TRANSFORM = "Transform"
TRAINING_TITLE = "Training"
EVALUATING_TITLE = "Evaluating"
AVERAGE = "Average"

# Default Parameters.
TRAINING = 1000
EVALUATING = 1000
SEED = 42
MAX_PROMPTS = 5
EXAMPLES = 10
DISTANCE_ERROR = 0.01
ANGLE_ERROR = 1
WAIT = 10

# Default bounding value.
BOUND = 2 * np.pi

# Handle timing out, as no single iteration should ever come close to reaching a second.
MAX_TIME = 1
ERROR_TIMED_OUT = "ERROR_TIMED_OUT"

# All fields for evaluations.
FIELDS = ["Success Rate (%)", "Failure Rate (%)", "Error Rate (%)", "Average Failure Distance",
          "Average Failure Angle (°)", "Average Elapsed Time (s)", "Generation Time (s)", "Mode", "Feedbacks Given",
          "Forwards Kinematics Calls", "Testing Calls", "Reasoning", "Functions", "API", "Cost ($)"]

# All numeric fields for evaluations.
NUMERIC = ["Success Rate (%)", "Failure Rate (%)", "Error Rate (%)", "Average Failure Distance",
           "Average Failure Angle (°)", "Average Elapsed Time (s)", "Generation Time (s)", "Feedbacks Given",
           "Forwards Kinematics Calls", "Testing Calls", "Cost ($)"]


def extract_method_call(s: str, forward_parameters: int, test_parameters: int, prioritize_forward: bool = False) -> str:
    """
    Extracts the last occurrence of either "FORWARD_KINEMATICS" or "TEST_SOLUTION" along with up to the specified
    number of parameters from the input string.
    :param s: The input plain-text string.
    :param forward_parameters: Maximum number of parameters for "FORWARD_KINEMATICS".
    :param test_parameters: Maximum number of parameters for "TEST_SOLUTION".
    :param prioritize_forward: On initial calls, we want to prioritize extracting forwards calls if multiple were made.
    :return: The method call with parameters as a single string, or an empty string if neither method is found.
    """
    # Define the method names and their corresponding max parameters.
    methods = {
        "FORWARD_KINEMATICS": forward_parameters,
        "TEST_SOLUTION": test_parameters
    }
    # See if we should prioritize forward kinematics calls.
    if prioritize_forward:
        # Compile a regex pattern to match only the forward kinematics calls.
        pattern = re.compile(r"(?<!\w)(FORWARD_KINEMATICS)\b")
        # Find all matches in the string.
        matches = list(pattern.finditer(s))
        # Otherwise, get the testing calls.
        if not matches:
            # Compile a regex pattern to match only the testing calls.
            pattern = re.compile(r"(?<!\w)(TEST_SOLUTION)\b")
            # Find all matches in the string.
            matches = list(pattern.finditer(s))
    # Otherwise, if not prioritized or no forward kinematics matches, search for all matches.
    else:
        # Compile a regex pattern to match either method name ensuring they are not part of larger words.
        pattern = re.compile(r"(?<!\w)(FORWARD_KINEMATICS|TEST_SOLUTION)\b")
        # Find all matches in the string.
        matches = list(pattern.finditer(s))
        # If no matches found, return empty string.
    if not matches:
        return ""
    # Select the last match.
    last_match = matches[-1]
    method_name = last_match.group(1)
    max_params = methods[method_name]
    # Get the substring starting right after the method name.
    start_pos = last_match.end()
    substring = s[start_pos:]
    # Define a regex pattern to match numeric values including positive and negative values with scientific notation.
    number_pattern = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
    # Find all numeric matches in the substring.
    numeric_params = number_pattern.findall(substring)
    # Extract up to the maximum number of parameters.
    extracted_params = numeric_params[:max_params]
    # If no numeric parameters found, return just the method name
    if not extracted_params:
        return method_name
    # Combine the method name and parameters into a single string
    return " ".join([method_name] + extracted_params)


def process_code(code: str) -> str:
    """
    Processes the input Python code string according to the specified rules:
    1. Replace print statements with pass.
    2. Remove empty if conditions with only pass.
    3. Remove empty loops with only pass.
    4. Remove empty functions with only pass and their calls.
    5. Remove trailing code after the last function.
    :param code: The code to clean.
    :return: The cleaned code.
    """

    class CodeTransformer(ast.NodeTransformer):
        """
        Helper class to transform code.
        """
        def __init__(self):
            """
            Initialize the code to flag that there are no empty functions and nothing has changed.
            """
            super().__init__()
            self.empty_functions = set()
            self.changed = False

        def replace_print_with_pass(self, node):
            """
            Replace print statements with passes.
            :param node: The node to replace.
            :return: The updated node.
            """
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name) and func.id == "print":
                    self.changed = True
                    return ast.Pass()
            return node

        def visit_Expr(self, node):
            """
            Visit an expression.
            :param node: The mode.
            :return: The node with all prints removed.
            """
            return self.replace_print_with_pass(node)

        def visit_Call(self, node):
            """
            Visit a call.
            :param node: The mode.
            :return: The node after visiting.
            """
            return self.generic_visit(node)

        def visit_FunctionDef(self, node):
            """
            Visit a function definition.
            :param node: The node.
            :return: The node after visiting.
            """
            self.generic_visit(node)
            if all(isinstance(stmt, ast.Pass) for stmt in node.body):
                self.empty_functions.add(node.name)
                self.changed = True
                # Remove the function.
                return None
            return node

        def visit_If(self, node):
            """
            Visit an if statement.
            :param node: The node.
            :return: The node after visiting.
            """
            self.generic_visit(node)
            if all(isinstance(stmt, ast.Pass) for stmt in node.body) and not node.orelse:
                self.changed = True
                # Remove the if statement.
                return None
            return node

        def visit_For(self, node):
            """
            Visit a for loop.
            :param node: The node.
            :return: The node after visiting.
            """
            self.generic_visit(node)
            if all(isinstance(stmt, ast.Pass) for stmt in node.body):
                self.changed = True
                # Remove the for loop.
                return None
            return node

        def visit_While(self, node):
            """
            Visit a while loop.
            :param node: The node.
            :return: The node after visiting.
            """
            self.generic_visit(node)
            if all(isinstance(stmt, ast.Pass) for stmt in node.body):
                self.changed = True
                # Remove the while loop.
                return None
            return node

        def remove_function_calls(self, node):
            """
            Removes standalone calls to empty functions.
            :param node: The node.
            :return: The node after visiting.
            """
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name) and func.id in self.empty_functions:
                    self.changed = True
                    # Remove the call.
                    return None
            return node

        def visit_Module(self, node):
            """
            Visit the entire module.
            :param node: The node.
            :return: The node after visiting.
            """
            self.generic_visit(node)
            # Remove trailing code after the last function.
            last_func = None
            for idx, stmt in enumerate(node.body):
                if isinstance(stmt, ast.FunctionDef):
                    last_func = idx
            if last_func is not None and last_func < len(node.body) - 1:
                node.body = node.body[:last_func + 1]
                self.changed = True
            return node
    # Parse the code into an AST.
    try:
        tree = ast.parse(code)
    except Exception as e:
        logging.info(f"Invalid Python code provided: {e}")
        return code
    transformer = CodeTransformer()
    # Step 1 to 5 with multiple passes.
    while True:
        transformer.changed = False
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)
        # Step 4: Remove calls to empty functions.
        if transformer.empty_functions:
            # Create a new transformer to remove calls to empty functions.
            call_remover = CodeTransformer()
            call_remover.empty_functions = transformer.empty_functions.copy()
            call_remover.visit = call_remover.remove_function_calls
            # Override to prevent further changes.
            call_remover.generic_visit = lambda node: node
            # noinspection PyArgumentList
            tree = call_remover.visit(tree)
            ast.fix_missing_locations(tree)
            transformer.changed = transformer.changed or call_remover.changed
        # If no changes in this pass, break.
        if not transformer.changed:
            break
        # Reset empty_functions for the next pass.
        transformer.empty_functions.clear()
    # Unparse the AST back to code.
    processed_code = ast.unparse(tree)
    return processed_code


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
                    with open(os.path.join(self.info, f"{lower + 1}-{upper + 1}.txt"), "w", encoding="utf-8",
                              errors="ignore") as file:
                        file.write(self.prepare_llm(lower, upper, True))
        logging.info(f"{self.name} | Loaded.")

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
                    failures = neat(failures / total * 100)
                    s = ("Success Rate (%),Failure Rate (%),Error Rate (%),Average Failure Distance,Average Failure "
                         "Angle (°),Average Elapsed Time (s),Generation Time (s),Mode,Feedbacks Given,Forwards "
                         f"Kinematics Calls,Testing Calls,Reasoning,Functions,API,Cost ($)\n{successes}%,{failures}%,0%"
                         f",{total_distance},{total_angle if orientation else 0}°,{total_time} s,0 s,,0,0,0,False,"
                         "False,False,$0")
                    # Save results.
                    os.makedirs(self.results, exist_ok=True)
                    path = os.path.join(self.results, f"{lower}-{upper}-{TRANSFORM if orientation else POSITION}.csv")
                    with open(path, "w", encoding="utf-8", errors="ignore") as file:
                        file.write(s)
        logging.info(f"{self.name} | Save Results | IKPy results saved.")

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
                         f"data loaded.")
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
                     f"data saved.")
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
             'link. The positions are from the "xyz" attribute and the orientations are the "rpy" attribute from each '
             'link\'s "origin" element parsed from the URDF.')
        if revolute > 0:
            s += (' The "Axis" column in the table represents the rotational axis of the revolute '
                  f"link{'s' if revolute > 1 else ''}; return {'their values' if revolute > 1 else 'the value'} in "
                  "radians")
            if limits:
                s += f" and {'their' if revolute > 1 else 'the'} limits are in radians"
            s += "."
        if prismatic > 0:
            s += (' The "Translation" column in the table represents the movement axis of the prismatic '
                  f"link{'s' if prismatic > 1 else ''}.")
        if fixed > 0:
            s += (f" The fixed link{'s do' if fixed > 1 else ' does'} not have any movement; do not return anything "
                  f"for these links.")
        s += (" Do not write any code to run or test the method, as this will be handled for you. Assume all targets "
              "given as inputs to the method will be reachable, and as such do not write code to check if the target is"
              " reachable. You may use any methods included in Python, NumPy, and SymPy to write your solution except "
              f"for any optimization methods.{additional}\n</INSTRUCTIONS>\n<DETAILS>\n{table}\n</DETAILS>\n<CODE>\n"
              "def inverse_kinematics(p: tuple[float, float, float]")
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
        s += (f') -> {ret}:\n    """\n    Gets the joint values needed to reach position "p"{reach}.\n    :param p: The'
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
                # noinspection PyTypeChecker
                values[i] = float(joints[index])
                index += 1
            # Otherwise, if not passed and there are bounds, set the midpoint.
            elif chain.links[i].bounds is not None:
                values[i] = np.average(chain.links[i].bounds)
            controlled.append(values[i])
        # Perform forward kinematics.
        try:
            links = chain.forward_kinematics(values, True)
        except Exception as e:
            logging.info(f"{self.name} | {lower + 1} to {upper + 1} | Forward kinematics | Joints = {controlled} | "
                         f"Forward kinematics failed; using zeros instead: {e}")
            for i in range(total):
                values[i] = 0
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

    def evaluate(self) -> None:
        """
        Get the results of individual solvers for this robot together.
        :return: Nothing.
        """
        # Nothing to evaluate if not valid.
        if not self.is_valid():
            logging.error(f"{self.name} | Evaluate | Robot not configured.")
            return None
        # Nothing to do if there are no results.
        results_root = os.path.join(RESULTS, self.name)
        if not os.path.exists(results_root):
            return None
        results = None
        # Parse the saved individual results from all solvers.
        paths = get_directories(results_root)
        for solver in paths:
            root = os.path.join(results_root, solver)
            files = get_files(root)
            # Get every chain this solver has done.
            for name in files:
                # The name needs to be properly formatted to match what it is for.
                path = os.path.join(root, name)
                parts = name.split("-")
                if len(parts) < 3:
                    logging.error(f"{self.name} | Evaluate | Result '{path}' not named properly.")
                    continue
                # Get the joints this is for.
                try:
                    lower = int(parts[0])
                    upper = int(parts[1])
                except Exception as e:
                    logging.error(f"{self.name} | Evaluate | Could not parse lower and upper from '{path}': {e}")
                    continue
                # Get what this was solving.
                solving = parts[2].replace(".csv", "")
                if solving != POSITION and solving != TRANSFORM:
                    logging.error(f"{self.name} | Evaluate | Could not parse solving from '{path}', must be either "
                                  f"'{POSITION}' or '{TRANSFORM}'.")
                    continue
                # Read the file.
                with open(path, "r", encoding="utf-8", errors="ignore") as file:
                    s = file.read()
                lines = s.splitlines()
                if len(lines) != 2:
                    logging.error(f"{self.name} | Evaluate | No result in '{path}'.")
                    continue
                # Ensure the right titles and fields are in it.
                expected = 15
                titles = lines[0].split(",")
                total_titles = len(titles)
                if total_titles != expected:
                    logging.error(f"{self.name} | Evaluate | Wrong number of titles in '{path}'; got {total_titles} but"
                                  f" expected {expected}.")
                    continue
                info = lines[1].split(",")
                total_results = len(info)
                if total_results != expected:
                    logging.error(f"{self.name} | Evaluate | Wrong number of results in '{path}'; got {total_results} "
                                  f"but expected {expected}.")
                    continue
                if total_titles != total_results:
                    logging.error(f"{self.name} | Evaluate | Titles and results in '{path}' do not match: "
                                  f"{total_titles} titles and {total_results} results.")
                    continue
                # Parse all data from the file.
                result = {}
                for i in range(expected):
                    title = titles[i]
                    data = info[i]
                    if title == "Success Rate (%)" or title == "Failure Rate (%)" or title == "Error Rate (%)":
                        try:
                            data = float(data.replace("%", ""))
                        except Exception as e:
                            logging.error(f"{self.name} | Evaluate | Could not parse percentage data at index {i + 1} "
                                          f"from '{path}': {e}")
                            result = None
                            break
                    elif title == "Average Failure Distance":
                        try:
                            data = float(data)
                        except Exception as e:
                            logging.error(f"{self.name} | Evaluate | Could not parse distance data at index {i + 1} "
                                          f"from '{path}': {e}")
                            result = None
                            break
                    elif title == "Average Failure Angle (°)":
                        try:
                            data = float(data.replace("°", ""))
                        except Exception as e:
                            logging.error(f"{self.name} | Evaluate | Could not parse angle data at index {i + 1} from "
                                          f"'{path}': {e}")
                            result = None
                            break
                    elif title == "Average Elapsed Time (s)" or title == "Generation Time (s)":
                        try:
                            data = float(data.replace(" s", ""))
                        except Exception as e:
                            logging.error(f"{self.name} | Evaluate | Could not parse time data at index {i + 1} from "
                                          f"'{path}': {e}")
                            result = None
                            break
                    elif title == "Feedbacks Given" or title == "Forwards Kinematics Calls" or title == "Testing Calls":
                        try:
                            data = int(data)
                        except Exception as e:
                            logging.error(f"{self.name} | Evaluate | Could not parse data at index {i + 1} from "
                                          f"'{path}': {e}")
                            result = None
                            break
                    elif title == "Reasoning" or title == "Functions" or title == "API":
                        try:
                            data = data == "True"
                        except Exception as e:
                            logging.error(f"{self.name} | Evaluate | Could not parse data at index {i + 1} from "
                                          f"'{path}': {e}")
                            result = None
                            break
                    elif title == "Cost ($)":
                        try:
                            data = float(data.replace("$", ""))
                        except Exception as e:
                            logging.error(f"{self.name} | Evaluate | Could not parse dollar data at index {i + 1} from "
                                          f"'{path}': {e}")
                            result = None
                            break
                    elif title != "Mode":
                        logging.error(f"{self.name} | Evaluate | Title '{title}' at index {i + 1} from '{path}' is not "
                                      f"valid.")
                        result = None
                        break
                    result[title] = data
                # Cache the data.
                if result is None:
                    continue
                result["Name"] = solver
                if results is None:
                    results = {}
                if lower not in results:
                    results[lower] = {}
                if upper not in results[lower]:
                    results[lower][upper] = {}
                if solving not in results[lower][upper]:
                    results[lower][upper][solving] = []
                results[lower][upper][solving].append(result)
        # If there were no results, there is nothing else to do.
        if results is None:
            return None
        # Otherwise, write the results for all solvers for each individual chain.
        for lower in results:
            for upper in results[lower]:
                for solving in results[lower][upper]:
                    # Sort the results to display the best solvers first.
                    results[lower][upper][solving] = sorted(
                        results[lower][upper][solving],
                        key=lambda x: (
                            -x["Success Rate (%)"],
                            x["Error Rate (%)"],
                            x["Average Failure Distance"],
                            x["Average Failure Angle (°)"],
                            x["Average Elapsed Time (s)"],
                            x["API"],
                            x["Cost ($)"],
                            x["Generation Time (s)"],
                            x["Mode"] == TRANSFER,
                            x["Mode"] == CUMULATIVE,
                            x["Mode"] == DYNAMIC,
                            x["Mode"] == EXTEND,
                            x["Mode"] == NORMAL,
                            x["Feedbacks Given"],
                            x["Forwards Kinematics Calls"],
                            x["Testing Calls"],
                            x["Reasoning"],
                            x["Functions"],
                            x["Name"]
                        )
                    )
                    # Format the results.
                    s = "Name," + ",".join(FIELDS)
                    for entry in results[lower][upper][solving]:
                        s += f"\n{entry['Name']}"
                        for field in FIELDS:
                            data = entry[field]
                            if field == "Success Rate (%)" or field == "Failure Rate (%)" or field == "Error Rate (%)":
                                data = f"{neat(data)}%"
                            elif field == "Average Failure Distance":
                                data = neat(data)
                            elif field == "Average Failure Angle (°)":
                                data = f"{neat(data)}°"
                            elif field == "Average Elapsed Time (s)" or field == "Generation Time (s)":
                                data = f"{neat(data)} s"
                            elif field == "Cost ($)":
                                data = f"${neat(data)}"
                            s += f",{data}"
                    path = os.path.join(results_root, f"{lower}-{upper}-{solving}.csv")
                    with open(path, "w", encoding="utf-8", errors="ignore") as file:
                        file.write(s)
        # Output to LaTeX as well for easy use.
        s = ""
        # Get positions first.
        for solving in [POSITION, TRANSFORM]:
            target = "Position" if solving == POSITION else "Position and Orientation"
            solving_started = False
            # Go from smallest to largest links.
            for length in range(self.joints):
                # Determine the last "first" joint for this size.
                length_started = False
                last = self.joints - length
                for lower in range(last):
                    # Get the upper joint index.
                    upper = lower + length
                    if solving not in results[lower][upper]:
                        continue
                    # Display the start of this category.
                    if not solving_started:
                        solving_started = True
                        t = f"\n\n\\subsection{{{target} Results}}"
                        if s == "":
                            s = f"\\begin{{landscape}}{t}"
                        else:
                            s += t
                    if len(results[lower][upper][solving]) < 0:
                        continue
                    # Display the title for this length.
                    if not length_started:
                        length_started = True
                        s += (f"\n\n\\subsubsection{{{num2words(length + 1).capitalize()} "
                              f"Degree{'' if length == 0 else 's'}-of-Freedom {target} Results}}")
                    # Determine what columns can be conditionally dropped.
                    needs_errors = False
                    needs_distance = False
                    needs_angle = False
                    needs_generated = False
                    needs_feedbacks = False
                    needs_forwards = False
                    needs_tests = False
                    needs_costs = False
                    for entry in results[lower][upper][solving]:
                        if entry["Error Rate (%)"] > 0:
                            needs_errors = True
                        if entry["Average Failure Distance"] > 0:
                            needs_distance = True
                        if entry["Average Failure Angle (°)"] > 0:
                            needs_angle = True
                        if entry["Generation Time (s)"] > 0:
                            needs_generated = True
                        if entry["Feedbacks Given"] > 0:
                            needs_feedbacks = True
                        if entry["Forwards Kinematics Calls"] > 0:
                            needs_forwards = True
                        if entry["Testing Calls"] > 0:
                            needs_tests = True
                        if entry["Cost ($)"] > 0:
                            needs_costs = True
                    # Build the table header
                    s += r"""

\begin{table}[H]
\tiny
\renewcommand{\arraystretch}{1.2}
\caption{"""
                    s += (f"Joint {num2words(lower + 1).capitalize()} {target} Results" if lower == upper else
                          f"Joints {num2words(lower + 1).capitalize()} to {num2words(upper + 1).capitalize()} {target} "
                          "Results")
                    s += r"""}
\begin{center}
\begin{tabular}{|c|c|c|c|"""
                    if needs_errors:
                        s += "c|"
                    if needs_distance:
                        s += "c|"
                    if needs_angle:
                        s += "c|"
                    if needs_generated:
                        s += "c|"
                    if needs_feedbacks:
                        s += "c|"
                    if needs_forwards:
                        s += "c|"
                    if needs_tests:
                        s += "c|"
                    if needs_costs:
                        s += "c|"
                    s += r"""}
    \hline
    \textbf{Model} & 
    \textbf{Mode} & 
    \makecell{\textbf{Success}\\\textbf{Rate (\%)}}"""
                    if needs_errors:
                        s += r""" &
    \makecell{\textbf{Error}\\\textbf{Rate (\%)}}"""
                    if needs_distance:
                        s += r""" &
    \makecell{\textbf{Avg. Fail}\\\textbf{Distance (mm)}}"""
                    if needs_angle:
                        s += r""" &
    \makecell{\textbf{Avg. Fail}\\\textbf{Angle (\textdegree)}}"""
                    s += r""" &
    \makecell{\textbf{Avg. Elapsed}\\\textbf{Time (ms)}}"""
                    if needs_generated:
                        s += r""" &
    \makecell{\textbf{Gen.}\\\textbf{Time (s)}}"""
                    if needs_feedbacks:
                        s += r""" &
    \textbf{Feedbacks}"""
                    if needs_forwards:
                        s += r""" &
    \makecell{\textbf{FK}\\\textbf{Calls}}"""
                    if needs_tests:
                        s += r""" &
    \makecell{\textbf{Test}\\\textbf{Calls}}"""
                    if needs_costs:
                        s += r""" &
    \textbf{Cost (\$)}"""
                    s += r""" \\
    \hline
"""
                    # Fill the rows
                    for entry in results[lower][upper][solving]:
                        s += f"    {entry['Name']} & "
                        if entry["Name"] == "IKPy":
                            s += "-"
                        elif entry["Mode"] == NORMAL:
                            s += "Direct"
                        else:
                            s += entry["Mode"]
                        t = f"{entry['Success Rate (%)']:.2f}".rstrip("0").rstrip(".")
                        s += f" & {t}\\%"
                        if needs_errors:
                            if entry["Success Rate (%)"] >= 100:
                                s += " & -"
                            else:
                                t = f"{entry['Error Rate (%)']:.2f}".rstrip("0").rstrip(".")
                                s += f" & {t}\\%"
                        if needs_distance:
                            if entry["Success Rate (%)"] >= 100 or entry['Error Rate (%)'] >= 100:
                                s += " & -"
                            else:
                                t = f"{entry['Average Failure Distance'] * 1000:.2f}".rstrip("0").rstrip(".")
                                s += f" & {t} mm"
                        if needs_angle:
                            if entry["Success Rate (%)"] >= 100 or entry['Error Rate (%)'] >= 100:
                                s += " & -"
                            else:
                                t = f"{entry['Average Failure Angle (°)']:.2f}".rstrip("0").rstrip(".")
                                s += f" & {t}\\textdegree"
                        if entry["Average Elapsed Time (s)"] == 0:
                            s += " & -"
                        elif str(entry["Average Elapsed Time (s)"]) == "inf":
                            s += " & Timeout"
                        else:
                            t = f"{entry['Average Elapsed Time (s)'] * 1000:.2f}".rstrip("0").rstrip(".")
                            s += f" & {t} ms"
                        if needs_generated:
                            if entry["Name"] == "IKPy":
                                s += " & -"
                            else:
                                t = f"{entry['Generation Time (s)']:.2f}".rstrip("0").rstrip(".")
                                s += f" & {t} s"
                        if needs_feedbacks:
                            if entry["Name"] == "IKPy":
                                s += " & -"
                            else:
                                s += f" & {entry['Feedbacks Given']}"
                        if needs_forwards:
                            if entry["Name"] == "IKPy":
                                s += " & -"
                            else:
                                s += f" & {entry['Forwards Kinematics Calls']}"
                        if needs_tests:
                            if entry["Name"] == "IKPy":
                                s += " & -"
                            else:
                                s += f" & {entry['Testing Calls']}"
                        if needs_costs:
                            if entry["Name"] == "IKPy":
                                s += " & -"
                            else:
                                t = f"{entry['Cost ($)']:.6f}".rstrip("0").rstrip(".")
                                s += f" & \\${t}"
                        s += r""" \\
    \hline
"""
                    s += r"""\end{tabular}
\label{Results-"""
                    s += f"{solving.capitalize()}-{lower + 1}-{upper + 1}"
                    s += r"""}
\end{center}
\end{table}"""
        s += r"""

\end{landscape}"""
        path = os.path.join(results_root, "Results.tex")
        with open(path, "w", encoding="utf-8", errors="ignore") as file:
            file.write(s)


class Solver:
    """
    Handle a solver attached to a robot.
    """

    def __init__(self, model: str, robot: Robot, inherited: list or None = None):
        """
        Load a solver.
        :param model: The name of the model.
        :param robot: The robot for the solver.
        :param inherited: Any solvers that can be inherited from.
        """
        self.code = None
        self.reasoning = False
        self.url = ""
        self.methods = False
        self.key = ""
        self.input_cost = -1
        self.output_cost = -1
        self.options = []
        self.api_name = None
        # Ensure the file exists.
        path = os.path.join(MODELS, f"{model}.txt")
        if not os.path.exists(path):
            logging.error(f"Model '{path}' does not exist.")
            self.model = ""
            self.robot = None
            self.interactions = os.path.join(INTERACTIONS, "_Invalid", "_Invalid")
            self.elapsed = os.path.join(ELAPSED, "_Invalid", "_Invalid")
            self.solutions = os.path.join(SOLUTIONS, "_Invalid", "_Invalid")
            self.results = os.path.join(RESULTS, "_Invalid", "_Invalid")
            self.tokens = os.path.join(TOKENS, "_Invalid", "_Invalid")
            return
        self.model = model
        self.robot = robot
        # If the robot is invalid, there is nothing to do.
        if self.robot is None:
            logging.error(f"{self.model} | Robot is null.")
            self.interactions = os.path.join(INTERACTIONS, "_Invalid", self.model)
            self.elapsed = os.path.join(ELAPSED, "_Invalid", self.model)
            self.solutions = os.path.join(SOLUTIONS, "_Invalid", self.model)
            self.results = os.path.join(RESULTS, "_Invalid", self.model)
            self.tokens = os.path.join(TOKENS, "_Invalid", self.model)
            return
        # Cache folders.
        self.interactions = os.path.join(INTERACTIONS, self.robot.name, self.model)
        self.elapsed = os.path.join(ELAPSED, self.robot.name, self.model)
        self.solutions = os.path.join(SOLUTIONS, self.robot.name, self.model)
        self.results = os.path.join(RESULTS, self.robot.name, self.model)
        self.tokens = os.path.join(TOKENS, self.robot.name, self.model)
        # Ensure the robot is valid.
        if not robot.is_valid():
            logging.error(f"{self.model} | {self.robot.name} | Robot is not valid.")
            return
        # Ensure itself is an option.
        self.options.append(self)
        # Read the models' file.
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            s = file.read()
        s = s.strip()
        lines = s.splitlines()
        total = len(lines)
        model_methods = None
        if total < 1:
            provider = ""
        else:
            # If there was information, use it to check if this is a reasoning model.
            reasoning = lines[0].strip().upper()
            if reasoning == "TRUE" or reasoning == "1":
                self.reasoning = True
                logging.info(f"{self.model} | {self.robot.name} | This is a reasoning model.")
            # Use a second line to indicate a provider.
            provider = lines[1].strip() if total >= 2 else ""
            # Get the input token cost.
            if total >= 3:
                input_cost = lines[2].replace("$", "").strip()
                try:
                    self.input_cost = max(float(input_cost), 0)
                except Exception as e:
                    logging.warning(f"{self.model} | {self.robot.name} | Could not parse input cost from "
                                    f"'{input_cost}': {e}")
                    self.input_cost = -1
            # Get the output token cost.
            if total >= 4:
                output_cost = lines[3].replace("$", "").strip()
                try:
                    self.output_cost = max(float(output_cost), 0)
                except Exception as e:
                    logging.warning(f"{self.model} | {self.robot.name} | Could not parse output cost from "
                                    f"'{output_cost}': {e}")
                    self.output_cost = -1
            # Get if this model supports methods.
            if total >= 5:
                model_methods = lines[4].strip().upper()
                model_methods = model_methods == "TRUE" or model_methods == "1"
            # Get a custom API name if one is provided
            if total >= 6:
                api_name = lines[5].strip()
                if api_name != "":
                    self.api_name = api_name
                    logging.info(f"{self.model} | {self.robot.name} | Using custom API name of '{self.api_name}'.")
        if not self.reasoning:
            logging.info(f"{self.model} | {self.robot.name} | This is not a reasoning model.")
        # If there are details, this indicates the provider of the model.
        if provider != "":
            # If the provider does not exist, indicate this.
            path = os.path.join(PROVIDERS, f"{provider}.txt")
            if not os.path.exists(path):
                logging.warning(f"{self.model} | {self.robot.name} | Provider '{path}' does not exist; setting this to "
                                "a chat interface without methods instead.")
            # Otherwise, load the provider.
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as file:
                    s = file.read()
                s = s.strip()
                # If there was no URL for the provider, indicate this.
                if s == "":
                    logging.warning(f"{self.model} | {self.robot.name} | Provider '{path}' does not have a URL; setting"
                                    " this to a chat interface without methods instead.")
                # Otherwise, set the URL.
                else:
                    lines = s.split()
                    self.url = lines[0].strip()
                    # The API paths should end with a "/".
                    if not self.url.endswith("/"):
                        self.url = f"{self.url}/"
                    logging.info(f"{self.model} | {self.robot.name} | Provider '{provider}' URL is '{self.url}'.")
                    # See if the API supports methods in addition to our per-model support.
                    if len(lines) > 1:
                        methods = lines[1].upper()
                        if methods == "TRUE" or methods == "1":
                            if model_methods is None or model_methods is True:
                                self.methods = True
                                logging.info(f"{self.model} | {self.robot.name} | Model and provider '{provider}' "
                                             f"support methods.")
                            else:
                                logging.info(f"{self.model} | {self.robot.name} | Provider '{provider}' support methods"
                                             f" but the model does not.")
                        else:
                            logging.info(f"{self.model} | {self.robot.name} | Provider '{provider}' does not support "
                                         "methods.")
                    else:
                        logging.info(f"{self.model} | {self.robot.name} | Provider '{provider}' does not support "
                                     "methods.")
        else:
            logging.info(f"{self.model} | {self.robot.name} | Chat interface model.")
        # If no path, remove costs, as non-API methods should not inherit or be inherited.
        if self.url == "":
            self.input_cost = -1
            self.output_cost = -1
            return
        # Otherwise, this is an API, so ensure costs are handled properly.
        if self.input_cost < 0 and self.output_cost < 0:
            logging.warning(f"{self.model} | {self.robot.name} | No costs defined; cannot be used for inheriting.")
        else:
            if self.input_cost < 0:
                logging.warning(f"{self.model} | {self.robot.name} | No output cost defined; using input cost of"
                                f"${self.input_cost}.")
                self.input_cost = self.output_cost
            elif self.output_cost < 0:
                logging.warning(f"{self.model} | {self.robot.name} | No input cost defined; using output cost of"
                                f"${self.output_cost}.")
                self.output_cost = self.input_cost
            # Set models that this API model can inherit from.
            self.set_inherited(inherited)
        # Load a key.
        path = os.path.join(KEYS, f"{provider}.txt")
        if not os.path.exists(path):
            logging.info(f"{self.model} | {self.robot.name} | No key at '{path}' for provider '{provider}'.")
            return
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            s = file.read()
        self.key = s.strip()
        if self.key == "":
            logging.warning(f"{self.model} | {self.robot.name} | No key specified in '{path}'.")
        else:
            logging.info(f"{self.model} | {self.robot.name} | Loaded API key.")

    def get_cost(self, lower: int = 0, upper: int = -1, orientation: bool = False,
                 mode: str = NORMAL) -> (float, int, int, int):
        """
        Get the cost of running a model.
        :param lower: The starting joint.
        :param upper: The ending joint.
        :param orientation: If this data cares about the orientation or not.
        :param mode: The mode by which the code was achieved.
        :return: The cost of running a model, the number of feedbacks given, and the number of each type of method call.
        """
        # Nothing to do if the solver is not valid.
        if not self.is_valid():
            logging.error(f"{self.model} | Get Cost | Solver is not valid.")
            return 0, 0, 0, 0
        # Ensure valid values.
        lower, upper = self.robot.validate_lower_upper(lower, upper)
        # If only one joint, can only solve in normal mode and for the position only.
        if lower == upper:
            mode = NORMAL
            orientation = False
        # Ensure the mode is valid.
        if mode not in [NORMAL, EXTEND, DYNAMIC, CUMULATIVE, TRANSFER]:
            logging.warning(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Get Cost | Mode '{mode}' "
                            f"not valid, using '{NORMAL}' instead.")
            mode = NORMAL
        # Get the cost.
        cost = 0
        solving = TRANSFORM if orientation else POSITION
        portion = f"{lower}-{upper}-{solving}-{mode}"
        root = os.path.join(self.tokens, portion)
        # Load the tokens used for each interaction.
        if os.path.exists(root):
            for tokens in get_files(root):
                # Read the file.
                path = os.path.join(root, tokens)
                with open(path, "r", encoding="utf-8", errors="ignore") as file:
                    s = file.read()
                # Extract the tokens from the file.
                lines = s.splitlines()
                if len(lines) < 2:
                    logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} "
                                  f"| Get Cost | '{path}' is not properly formatted.")
                    continue
                data = lines[1].split(",")
                if len(data) < 2:
                    logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} "
                                  f"| Get Cost | Could not parse '{path}'.")
                    continue
                try:
                    inputs = int(data[0])
                    outputs = int(data[1])
                except Exception as e:
                    logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} "
                                  f"| Get Cost | Could not extract tokens from '{path}': {e}")
                    continue
                # Add the cost for each response.
                cost += (inputs * self.input_cost) + (outputs * self.output_cost)
        # Get all types of interactions.
        feedbacks = 0
        forwards = 0
        tests = 0
        root = os.path.join(self.interactions, portion)
        if os.path.exists(root):
            for file in get_files(root):
                if MESSAGE_FEEDBACK in file:
                    feedbacks += 1
                elif MESSAGE_FORWARD in file:
                    forwards += 1
                elif MESSAGE_DONE in file:
                    tests += 1
        # Load any inherited data.
        path = os.path.join(self.interactions, portion, f"{INHERITED}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as file:
                s = file.read()
            lines = s.splitlines()
            for line in lines:
                info = line.split("|")
                # Ensure the right amount of info is present.
                if len(info) < 5:
                    logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} "
                                  f"| Get Cost | Inheriting data in '{path}' not properly formatted.")
                    continue
                # Ensure the indicated model can be inherited.
                sub_model = info[0]
                found = False
                for solver in self.options:
                    if solver.model == sub_model:
                        sub_model = solver
                        found = True
                        break
                if not found:
                    logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} "
                                  f"| Get Cost | Model to inherit '{sub_model}' in '{path}' is not an option.")
                    continue
                # Parse lower and upper bounds.
                try:
                    sub_lower = int(info[1])
                    sub_upper = int(info[2])
                except Exception as e:
                    logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} "
                                  f"| Get Cost | Could not extract joint parts from '{path}': {e}")
                    continue
                # Parse what it was solving for.
                sub_solving = info[3]
                if sub_solving not in [POSITION, TRANSFORM]:
                    logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} "
                                  f"| Get Cost | Invalid solving for of '{sub_solving}' in '{path}'.")
                    continue
                # Parse the mode it was done in.
                sub_mode = info[4]
                if sub_mode not in [NORMAL, EXTEND, DYNAMIC, CUMULATIVE, TRANSFER]:
                    logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} "
                                  f"| Get Cost | Invalid mode of '{sub_mode}' in '{path}'.")
                    continue
                # Add the cost of this inherited model.
                s_cost, s_feedbacks, s_forwards, s_tests = sub_model.get_cost(sub_lower, sub_upper,
                                                                              sub_solving == TRANSFORM, sub_mode)
                cost += s_cost
                feedbacks += s_feedbacks
                forwards += s_forwards
                tests += s_tests
        # Return the overall cost.
        return cost, feedbacks, forwards, tests

    def set_inherited(self, inherited: list or None = None) -> None:
        """
        Set the models an API-based model can inherit from.
        :param inherited: The models to potentially inherit.
        :return: Nothing.
        """
        # Nothing to do if the solver is not valid.
        if not self.is_valid():
            self.options = []
            logging.error(f"{self.model} | Set Inherited | Solver is not valid.")
            return None
        # If nothing is passed or this does not use an API with a valid cost, use only itself as an option.
        if self.url == "":
            self.options = [self]
            logging.info(f"{self.model} | {self.robot.name} | Set Inherited | Only API models can inherit.")
            return None
        if self.input_cost < 0 or self.output_cost < 0:
            self.options = [self]
            logging.info(f"{self.model} | {self.robot.name} | Set Inherited | Cannot inherit as no costs defined.")
            return None
        if inherited is None or len(inherited) < 1:
            self.options = [self]
            logging.info(f"{self.model} | {self.robot.name} | Set Inherited | Nothing to inherit.")
            return None
        # Get only valid solvers which were passed.
        options = []
        for solver in inherited:
            # Check if they are solvers.
            if not isinstance(solver, Solver):
                logging.warning(f"{self.model} | {self.robot.name} | Set Inherited | Element '{solver}' is not a "
                                "solver.")
                continue
            # Ensure the solver to inherit is correct.
            if not solver.is_valid():
                logging.warning(f"{self.model} | {self.robot.name} | Set Inherited | Element '{solver}' is not valid.")
                continue
            # If this is not a reasoning model, do not inherit reasoning models.
            if not self.reasoning and solver.reasoning:
                logging.info(f"{self.model} | {self.robot.name} | Set Inherited | Non-reasoning models cannot inherit "
                             "reasoning models.")
                continue
            # If this is a reasoning and checking a non-reasoning, we are good, otherwise more checks.
            if not self.reasoning or solver.reasoning:
                # Check that they have a valid cost.
                if solver.input_cost < 0 or solver.output_cost < 0:
                    logging.info(f"{self.model} | {self.robot.name} | Set Inherited | Can only inherit API methods "
                                 "which have costs.")
                    continue
                # Can only inherit models with better costs.
                if self.input_cost < solver.input_cost:
                    logging.info(f"{self.model} | {self.robot.name} | Set Inherited | Can only inherit API methods with"
                                 " better input costs.")
                    continue
                if self.output_cost < solver.output_cost:
                    logging.info(f"{self.model} | {self.robot.name} | Set Inherited | Can only inherit API methods with"
                                 " better output costs.")
                    continue
            # Make sure this is for the proper robot.
            if self.robot.name != solver.robot.name:
                logging.info(f"{self.model} | {self.robot.name} | Set Inherited | Can only inherit the same robot.")
                continue
            # Check it is not this.
            if solver == self or solver.model == self.model:
                logging.info(f"{self.model} | {self.robot.name} | Set Inherited | Cannot manually inherit from self.")
                continue
            # Check this element has not yet been added.
            if solver in options:
                logging.info(f"{self.model} | {self.robot.name} | Set Inherited | Can only add one instance of each to "
                             "inherit.")
                continue
            # Check for name duplicates.
            match = False
            for option in options:
                if solver.model == option.model:
                    logging.info(f"{self.model} | {self.robot.name} | Set Inherited | Can only add one instance of each"
                                 " to inherit.")
                    match = True
                    break
            if match:
                continue
            # If all checks have been passed, we can inherit it.
            options.append(solver)
        # Sort by non-reasoning, output cost, input cost, and lastly name.
        if len(options) > 1:
            options = sorted(options, key=lambda x: (x.reasoning, x.output_cost, x.input_cost, x.model))
        # Lastly, using itself is the final option.
        options.append(self)
        # Save the options.
        self.options = options

    def perform(self, orientation: bool = False, mode: str = NORMAL, max_length: int = 0, run: bool = False,
                wait: int = WAIT) -> bool:
        """
        Perform solver logic.
        :param orientation: If we want to solve for orientation in addition to position.
        :param mode: The highest mode we want to run API calls with
        :param max_length: The maximum chain length to run.
        :param run: If API calls should be run.
        :param wait: How long to wait between API calls.
        :return: True if there were no API errors, false otherwise.
        """
        # Nothing to load if the solver is not valid.
        if not self.is_valid():
            logging.error(f"{self.model} | Perform | Solver is not valid.")
            return True
        # Set the solution types we want to solve for.
        orientation = [False, True] if orientation else [False]
        # Get the modes to run in.
        if mode == TRANSFER:
            mode = [NORMAL, EXTEND, DYNAMIC, CUMULATIVE, TRANSFER]
        elif mode == CUMULATIVE:
            mode = [NORMAL, EXTEND, DYNAMIC, CUMULATIVE]
        elif mode == DYNAMIC:
            mode = [NORMAL, EXTEND, DYNAMIC]
        elif mode == EXTEND:
            mode = [NORMAL, EXTEND]
        else:
            mode = [NORMAL]
        # Get the maximum length of chains to run.
        if max_length < 1:
            max_length = self.robot.joints
        # Loop all possible combinations.
        for current_orientation in [False, True]:
            # Solve smaller chains first so their solutions can be extended.
            for length in range(self.robot.joints):
                # Determine the last "first" joint for this size.
                last = self.robot.joints - length
                for lower in range(last):
                    # Get the upper joint index.
                    upper = lower + length
                    for current_mode in [NORMAL, EXTEND, DYNAMIC, CUMULATIVE, TRANSFER]:
                        # No solving for orientation with just one link and can only do normal prompting.
                        if length == 0 and (current_orientation or current_mode != NORMAL):
                            break
                        # Transfer mode is only for orientation.
                        if not current_orientation and current_mode == TRANSFER:
                            continue
                        # Handle the interaction as much as possible.
                        while True:
                            # Get the messages to send to the LLM.
                            messages = self.handle_interactions(lower, upper, current_orientation, current_mode)
                            # If there are no messages, or we should not call the LLM due to parameters, stop.
                            if (messages is None or len(messages) < 1 or self.url == "" or not run
                                    or current_orientation not in orientation or current_mode not in mode
                                    or length >= max_length):
                                break
                            # Run the API if all checks were passed.
                            if not self.run_api(lower, upper, current_orientation, current_mode, messages):
                                logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | "
                                              f"{TRANSFORM if current_orientation else POSITION} | {current_mode} | "
                                              "Stopping API calls as there was an error.")
                                return False
                            # On a success or a continue error, wait for the next API call if needed.
                            if run and wait > 1:
                                logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | "
                                             f"{TRANSFORM if current_orientation else POSITION} | {current_mode} | "
                                             f"Waiting for {wait} second{'' if wait == 1 else 's'} before next API "
                                             f"call.")
                                time.sleep(wait)
        # Return if everything was successful or not.
        return True

    def run_api(self, lower: int = 0, upper: int = -1, orientation: bool = False, mode: str = NORMAL,
                messages: list[dict[str, str or bool]] or None = None) -> bool:
        """
        Handle interacting with OpenAI-capable APIs.
        :param lower: The starting joint.
        :param upper: The ending joint.
        :param orientation: If this data cares about the orientation or not.
        :param mode: The mode by which the code was achieved.
        :param messages: The messages to send to the LLM.
        :return: True if the API was successfully queried, false otherwise.
        """
        # Nothing to do if the solver is not valid.
        if not self.is_valid():
            logging.error(f"{self.model} | Run API | Solver is not valid.")
            return False
        # Ensure valid values.
        lower, upper = self.robot.validate_lower_upper(lower, upper)
        # If only one joint, can only solve in normal mode and for the position only.
        if lower == upper:
            if mode != NORMAL:
                logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Run API | Must be in "
                              f"'{NORMAL}' mode for one joint.")
                return False
            if orientation:
                logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Run API | Can't do "
                              "orientation for one joint.")
                return False
        # Ensure the mode is valid.
        if mode not in [NORMAL, EXTEND, DYNAMIC, CUMULATIVE, TRANSFER]:
            logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Run API | Mode '{mode}' is "
                          "not valid.")
            return False
        solving = TRANSFORM if orientation else POSITION
        # Nothing to do if there are no messages.
        if messages is None or len(messages) < 1:
            logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Run "
                          "API | No messages to give to the LLM.")
            return False
        # Check all messages.
        total = len(messages)
        for i in range(total):
            # Ensure the element exists.
            if messages[i] is None:
                logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | "
                              f"Run API | Message at index {i} not set.")
                return False
            # Ensure the needed fields exist.
            if "Type" not in messages[i]:
                logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | "
                              f'Run API | Message at index {i} does not have a "Type" field.')
                return False
            if "Message" not in messages[i]:
                logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | "
                              f'Run API | Message at index {i} does not have a "Message" field.')
                return False
            # Starting with the first message as a prompt, messages alternate between it and responses.
            is_prompt = i % 2 == 0
            if is_prompt and messages[i]["Type"] == RESPONSE:
                logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | "
                              f"Run API | Message at index {i} expected to be a prompt but was not.")
                return False
            elif not is_prompt and messages[i]["Type"] != RESPONSE:
                logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | "
                              f"Run API | Message at index {i} expected to be a response but was not.")
                return False
            # Ensure no messages are empty.
            messages[i]["Message"] = messages[i]["Message"].strip()
            if messages[i]["Message"] == "":
                logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | "
                              f"Run API | Message at index {i} is empty.")
                return False
            # Reformat the messages to ensure they are ready for the API.
            messages[i] = {"role": "user" if is_prompt else "assistant", "content": messages[i]["Message"]}
        # The last message must be a prompt for the LLM.
        if messages[-1]["role"] != "user":
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Run API"
                         " | Last message is not a prompt.")
            return False
        logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Run API | "
                     f"Messages: {messages}")
        core = f"{lower}-{upper}-{solving}-{mode}"
        # If this can use methods, set them.
        if self.methods:
            tools = []
            tool_choice = "auto"
            # Build the command to use forward kinematics.
            forwards = {
                "type": "function",
                "function": {
                    "name": "FORWARD_KINEMATICS",
                    "description": "Test the forward kinematics of the robot.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
            # Add every joint.
            joints = upper - lower + 1
            required = []
            for i in range(joints):
                num = i + 1
                forwards["function"]["parameters"]["properties"][f"joint{num}"] = {
                    "type": "number",
                    "description": f"The value to set joint {num} to."
                }
                required.append(f"joint{num}")
            forwards["function"]["parameters"]["required"] = required
            tools.append(forwards)
            # If a solution exists, we can add the command to test it.
            if os.path.exists(os.path.join(self.solutions, f"{core}.py")):
                tests = {
                    "type": "function",
                    "function": {
                        "name": "TEST_SOLUTION",
                        "description": "Test your current solution."
                    }
                }
                if orientation:
                    tests["function"]["parameters"] = {
                        "type": "object",
                        "properties": {
                            "positionX": {
                                "type": "number",
                                "description": "The X position to reach."
                            },
                            "positionY": {
                                "type": "number",
                                "description": "The Y position to reach."
                            },
                            "positionZ": {
                                "type": "number",
                                "description": "The Z position to reach."
                            },
                            "orientationX": {
                                "type": "number",
                                "description": "The X orientation to reach in radians."
                            },
                            "orientationY": {
                                "type": "number",
                                "description": "The Y orientation to reach in radians."
                            },
                            "orientationZ": {
                                "type": "number",
                                "description": "The Z orientation to reach in radians."
                            }
                        },
                        "required": ["positionX", "positionY", "positionZ", "orientationX", "orientationY",
                                     "orientationZ"]
                    }
                else:
                    tests["function"]["parameters"] = {
                        "type": "object",
                        "properties": {
                            "positionX": {
                                "type": "number",
                                "description": "The X position to reach."
                            },
                            "positionY": {
                                "type": "number",
                                "description": "The Y position to reach."
                            },
                            "positionZ": {
                                "type": "number",
                                "description": "The Z position to reach."
                            },
                        },
                        "required": ["positionX", "positionY", "positionZ"]
                    }
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Run API"
                         f" | Tools: {tools}")
        else:
            tools = NOT_GIVEN
            tool_choice = NOT_GIVEN
        # Build the API client.
        client = OpenAI(api_key=self.key, base_url=self.url)
        # Try to call the API.
        while True:
            try:
                start_time = time.perf_counter()
                # Use the API-specific name if one exists.
                completion = client.chat.completions.create(
                    model=self.model if self.api_name is None else self.api_name, messages=messages, tools=tools,
                    tool_choice=tool_choice, seed=SEED, temperature=0, n=1,
                    reasoning_effort="high" if self.reasoning else NOT_GIVEN
                )
                elapsed = time.perf_counter() - start_time
            except Exception as e:
                logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | "
                              f"Run API | Error calling the API: {e}")
                return False
            # Log the raw response.
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Run API"
                         f" | Response | {completion}")
            # See if there is an error directly indicated.
            if hasattr(completion, "error") and completion.error is not None:
                s = f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Run API"
                if "code" in completion.error:
                    if int(completion.error["code"]) == 402:
                        logging.warning(f"{s} | Out of credits.")
                        return False
                    s += f" | Code: {completion.error['code']}"
                if "message" in completion.error:
                    s += f" | Message: {completion.error['message']}"
                # If the error did not result in any tokens being used, we can freely try again.
                if hasattr(completion, "usage") and completion.usage is None:
                    logging.warning(f"{s} | No usage so will attempt again.")
                    continue
                # Otherwise, stop.
                logging.error(s)
                return False
            break
        # Get the response message.
        if completion.choices is None or len(completion.choices) < 1:
            logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Run "
                          "API | No content in the response.")
            return False
        # See if there was an error that caused the API to stop.
        response = completion.choices[0]
        if response.finish_reason is None:
            if response.message is not None and response.message.refusal is not None:
                reason = f": {response.message.refusal}"
            else:
                reason = "."
            logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Run "
                          f"API | No finish reason given in the response{reason}")
            return False
        # Check if we ran out of tokens.
        finish_reason = response.finish_reason.upper()
        if finish_reason == "LENGTH":
            if response.message is not None and response.message.refusal is not None:
                reason = f": {response.message.refusal}"
            else:
                reason = "."
            logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Run "
                          f"API | Stopped as ran out of tokens{reason}")
            return False
        # Check if the content filter flagged this.
        if finish_reason == "CONTENT_FILTER":
            if response.message is not None and response.message.refusal is not None:
                reason = f": {response.message.refusal}"
            else:
                reason = "."
            logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Run "
                          f"API | Stopped as this was flagged by the content filter{reason}")
            return False
        # Check if no message was returned.
        if response.message is None:
            logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Run "
                          "API | No message in the response.")
            return False
        # Get the message in the response.
        message = response.message
        # Check if there is a reason for refusing to create the proper message.
        if message.refusal is not None and message.refusal != "":
            logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Run "
                          f"API | Refused to return a message: {message.refusal}")
            return False
        # Handle if this was a regular text response.
        if finish_reason == "STOP":
            # If for some reason there was no content, there was some undocumented error.
            if message.content is None or len(message.content) < 1:
                logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | "
                              "Run API | No content returned in the message.")
                return False
            # Get the message.
            s = message.content.strip().replace("\u2003", " ")
            if len(s) < 1:
                logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | "
                              "Run API | Returned content was whitespace.")
                return False
        # Handle if this was a tools call, also checking the older deprecated function call name.
        elif finish_reason == "TOOL_CALLS" or finish_reason == "FUNCTION_CALL":
            # Ensure there is a call to a function.
            if message.tool_calls is None or len(message.tool_calls) < 1:
                logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | "
                              "Run API | No tools calls returned in the message.")
                return False
            # Get the first call as there should only be one anyway.
            call = message.tool_calls[0]
            # If the object is not set, there was some undocumented error.
            if call is None:
                logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | "
                              "Run API | Tool call not set.")
                return False
            # Get the function object.
            if call.function is None:
                logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | "
                              "Run API | No function in the tools response.")
                return False
            # We only need to focus on the call.
            call = call.function
            # If the name was not present, there was some undocumented error.
            if call.name is None:
                logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | "
                              "Run API | No called function name in response.")
                return False
            # Get the name to save.
            s = call.name
            # If the arguments were not present, we will assume they will be empty.
            if call.arguments is not None:
                # Try to extract the arguments.
                try:
                    arguments = json.loads(call.arguments)
                except Exception as e:
                    logging.error(
                        f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | "
                        f"Run API | Could not parse the arguments: {e}")
                    return False
                # Sort the arguments.
                arguments = dict(sorted(arguments.items(), key=lambda item: (item[0], item[1])))
                # Add the arguments to save.
                for argument in arguments:
                    s += f" {arguments[argument]}"
        # Otherwise, the finish reason is an undocumented error.
        else:
            logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Run "
                          f"API | Unknown finish reason of '{finish_reason}' returned.")
            return False
        # Get the number of tokens used to handle this interaction.
        if completion.usage is None:
            logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Run "
                          f"API | No usage statistics returned.")
            return False
        if completion.usage.prompt_tokens is None:
            logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Run "
                          f"API | No prompt tokens returned.")
            return False
        input_tokens = completion.usage.prompt_tokens
        if completion.usage.completion_tokens is None:
            logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Run "
                          f"API | No completion tokens returned.")
            return False
        output_tokens = completion.usage.completion_tokens
        # Save the interaction.
        root = os.path.join(self.interactions, core)
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, f"{total}-{RESPONSE}.txt"), "w", encoding="utf-8", errors="ignore") as file:
            file.write(s)
        # Save the tokens.
        s = f"Input Tokens,Output Tokens\n{input_tokens},{output_tokens}"
        root = os.path.join(self.tokens, core)
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, f"{total}.csv"), "w", encoding="utf-8", errors="ignore") as file:
            file.write(s)
        # Save the elapsed time.
        s = f"Generation Time (s)\n{elapsed} s"
        root = os.path.join(self.elapsed, core)
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, f"{total}.csv"), "w", encoding="utf-8", errors="ignore") as file:
            file.write(s)
        logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Run API | "
                     "Response received.")
        return True

    def get_best(self, lower: int = 0, upper: int = -1, orientation: bool = False) -> (Any or None, str, float):
        """
        Get the best code for a certain size.
        :param lower: The starting joint.
        :param upper: The ending joint.
        :param orientation: If this data cares about the orientation or not.
        :return: The best solver possible, the mode it was solved in, and the cost it took to solve.
        """
        # Nothing to do if the solver is not valid.
        if not self.is_valid():
            logging.error(f"{self.model} | Get Best | Solver is not valid.")
            return None, NORMAL, 0
        # Ensure valid values.
        lower, upper = self.robot.validate_lower_upper(lower, upper)
        # If only one joint, can only solve for the position only.
        if lower == upper:
            orientation = False
        # Determine the best sub-option for this across all possible modes.
        best = None
        best_mode = NORMAL
        best_cost = 0
        for mode_option in [NORMAL, EXTEND, DYNAMIC, CUMULATIVE, TRANSFER]:
            for solver_option in self.options:
                if solver_option.code_successful(lower, upper, orientation, mode_option):
                    # If this is the first successful one, use it.
                    if best is None:
                        best = solver_option
                        best_mode = mode_option
                        best_cost, feedbacks, forwards, tests = solver_option.get_cost(lower, upper, orientation,
                                                                                       mode_option)
                    # Otherwise, use it if it is a better cost.
                    else:
                        cost, feedbacks, forwards, tests = solver_option.get_cost(lower, upper, orientation,
                                                                                  mode_option)
                        if cost >= best_cost:
                            continue
                        best = solver_option
                        best_mode = mode_option
                        best_cost = cost
        return best, best_mode, best_cost

    def handle_interactions(self, lower: int = 0, upper: int = -1, orientation: bool = False,
                            mode: str = NORMAL) -> list[dict[str, str or bool]] or None:
        """
        Handle determining what the next messages should be.
        :param lower: The starting joint.
        :param upper: The ending joint.
        :param orientation: If this data cares about the orientation or not.
        :param mode: The mode by which the code was achieved.
        :return: The interactions with the LLM or none if there is an error or the interacting is done.
        """
        # Nothing to do if the solver is not valid.
        if not self.is_valid():
            logging.error(f"{self.model} | Handle Interactions | Solver is not valid.")
            return None
        # Ensure valid values.
        lower, upper = self.robot.validate_lower_upper(lower, upper)
        # If only one joint, can only solve in normal mode and for the position only.
        if lower == upper:
            mode = NORMAL
            orientation = False
        # Can only transfer to orientation mode.
        if not orientation and mode == TRANSFER:
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Can only use transfer mode "
                         "for orientation solving.")
            return None
        # Ensure the mode is valid.
        if mode not in [NORMAL, EXTEND, DYNAMIC, CUMULATIVE, TRANSFER]:
            logging.warning(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Handle Interactions | "
                            f"Mode '{mode}' not valid, using '{NORMAL}' instead.")
            mode = NORMAL
        solving = TRANSFORM if orientation else POSITION
        # Get all interactions.
        root = os.path.join(self.interactions, f"{lower}-{upper}-{solving}-{mode}")
        # We do not want the "Details" file as this is simply for results analysis.
        interactions = [s for s in get_files(root) if INHERITED not in s] if os.path.exists(root) else []
        total = len(interactions)
        # Build the conversation history.
        history = []
        replies = 0
        for i in range(total):
            searching = f"{i}-"
            current = None
            for interaction in interactions:
                if interaction.startswith(searching):
                    current = interaction
                    break
            if current is None:
                logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | "
                              f"Handle Interactions | No interaction starts with '{searching}'.")
                return None
            # Get the type of message this was.
            if MESSAGE_PROMPT in current:
                current_type = MESSAGE_PROMPT
            elif MESSAGE_FEEDBACK in current:
                current_type = MESSAGE_FEEDBACK
                replies += 1
            elif MESSAGE_TEST in current:
                current_type = MESSAGE_TEST
                replies += 1
            elif MESSAGE_FORWARD in current:
                current_type = MESSAGE_FORWARD
                replies += 1
            elif MESSAGE_DONE in current:
                current_type = MESSAGE_DONE
            elif RESPONSE in current:
                current_type = RESPONSE
            else:
                logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | "
                              f"Handle Interactions | No valid message type in '{searching}'.")
                return None
            path = os.path.join(root, interactions[i])
            with open(path, "r", encoding="utf-8", errors="ignore") as file:
                s = file.read()
            history.append({"Type": current_type, "Message": s})
        # Create the initial message.
        total = len(history)
        if total < 1:
            s = self.prepare_llm(lower, upper, orientation, mode)
            if s != "":
                os.makedirs(root, exist_ok=True)
                with open(os.path.join(root, f"0-{MESSAGE_PROMPT}.txt"), "w", encoding="utf-8",
                          errors="ignore") as file:
                    file.write(s)
                logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | "
                             f"Handle Interactions | Initial prompt generated.")
                return [{"Type": MESSAGE_PROMPT, "Message": s}]
            # If no prompt was made, there is nothing to return.
            return None
        # If the last interaction was a message for the LLM, load it.
        last = history[-1]
        if last["Type"] == MESSAGE_DONE:
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Done.")
            return None
        # See if we should be done.
        are_done = replies >= MAX_PROMPTS
        # If this is a message for the model, we should try to load it.
        if last["Type"] != RESPONSE:
            # See if we are done.
            if are_done:
                logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | "
                             f"Handle Interactions | A message for the model exists but {MAX_PROMPTS} feedback"
                             f"{' has' if MAX_PROMPTS == 1 else 's have'} been used; stopping.")
                with open(os.path.join(root, f"{total}-{MESSAGE_DONE}.txt"), "w", encoding="utf-8",
                          errors="ignore") as file:
                    file.write(f"A message for the model exists but {MAX_PROMPTS} feedback"
                               f"{' has' if MAX_PROMPTS == 1 else 's have'} been used; stopping.")
                return None
            # Otherwise, give it the message.
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Handle "
                         f"Interactions | Messages loaded.")
            return history
        s = last["Message"]
        code_path = os.path.join(self.solutions, f"{lower}-{upper}-{solving}-{mode}.py")
        # Otherwise, it was a response from the LLM, so prepare the next message by parsing it.
        codes = re.findall(r"```python\s*([\s\S]*?)```", s, re.IGNORECASE)
        # If no codes were returned, this means it was a command response or invalid, so determine this.
        total_codes = len(codes)
        if total_codes < 1:
            # See if we are done.
            if are_done:
                logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | "
                             f"Handle Interactions | A command was requested but {MAX_PROMPTS} feedback"
                             f"{' has' if MAX_PROMPTS == 1 else 's have'} been used; stopping.")
                with open(os.path.join(root, f"{total}-{MESSAGE_DONE}.txt"), "w", encoding="utf-8",
                          errors="ignore") as file:
                    file.write(f"A command was requested but {MAX_PROMPTS} feedback"
                               f"{' has' if MAX_PROMPTS == 1 else 's have'} been used; stopping.")
                return None
            # Extract the method call.
            forwards_expected = upper - lower + 1
            testing_expected = 6 if orientation else 3
            # If no solution exists yet, we should prioritize extracting forward kinematics calls.
            line = extract_method_call(s, forwards_expected, testing_expected, not os.path.exists(code_path)).split()
            parts = len(line)
            if parts > 0:
                # Handle if this is a forward kinematics call.
                if line[0] == "FORWARD_KINEMATICS":
                    received = parts - 1
                    # Ensure the proper number of joints were given.
                    if received != forwards_expected:
                        logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | "
                                     f"{mode} | Handle Interactions | Forward kinematics call had wrong number of "
                                     "joints.")
                        s = ("<ERROR>\nResponded with the wrong number of joints to call forward kinematics - Responded"
                             f" with {received} but expected {forwards_expected}.\n</ERROR>")
                    else:
                        try:
                            # Parse the joints.
                            joints = []
                            for j in range(1, parts):
                                joints.append(float(line[j]))
                            # Run the forward kinematics and format results.
                            positions, orientations = self.robot.forward_kinematics(lower, upper, joints)
                            headers = ["Link", "Position", "Orientation"]
                            data = []
                            chain = self.robot.chains[lower][upper]
                            num = len(chain.links)
                            for j in range(num):
                                data.append([chain.links[j].name, neat(positions[j]), neat(orientations[j])])
                            s = (f"<FORWARD_KINEMATICS>\n{tabulate(data, headers, tablefmt='presto')}\n"
                                 "</FORWARD_KINEMATICS>")
                            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | "
                                         f"{mode} | Handle Interactions | Performed forward kinematics.")
                        except Exception as e:
                            # Indicate if the joints could not be parsed.
                            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | "
                                         f"{mode} | Handle Interactions | Forward kinematics did not respond with valid"
                                         f" joints: {e}")
                            s = ("<ERROR>\nCould not parse joint values to call forward kinematics; ensure they are all"
                                 " floats.\n</ERROR>")
                    os.makedirs(root, exist_ok=True)
                    with open(os.path.join(root, f"{total}-{MESSAGE_FORWARD}.txt"), "w", encoding="utf-8",
                              errors="ignore") as file:
                        file.write(s)
                    history.append({"Type": MESSAGE_FORWARD, "Message": s})
                    return history
                # Handle if this is a solution testing call.
                elif line[0] == "TEST_SOLUTION":
                    received = parts - 1
                    # If there is no solution to begin with, there is nothing to do.
                    if not os.path.exists(code_path):
                        logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | "
                                     f"{mode} | Handle Interactions | No solution to test.")
                        s = ("<ERROR>\nYou have not yet provided a solution to the code for testing. Please provided "
                             "one before calling this function.\n</ERROR>")
                    # Indicate if the wrong number of parameters were received.
                    elif received != testing_expected:
                        logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | "
                                     f"{mode} | Handle Interactions | Test solution call had wrong number of "
                                     "parameters.")
                        s = ("<ERROR>\nResponded with the wrong number of parameters to test your solution - Responded "
                             f"with {received} but expected {testing_expected}.\n</ERROR>")
                    else:
                        try:
                            # Parse the position and orientation to reach.
                            target_position = []
                            for j in range(3):
                                target_position.append(float(line[j]))
                            if orientation:
                                target_orientation = []
                                for j in range(3, 6):
                                    target_orientation.append(float(line[j]))
                            else:
                                target_orientation = None
                            # Run the code.
                            joints, e, error = self.run_code(lower, upper, mode, target_position, target_orientation)
                            # Indicate if there was an error.
                            expected = upper - lower + 1
                            if error is not None:
                                if error == ERROR_TIMED_OUT:
                                    s = ("<ERROR>Code timed out after taking too long. Ensure your solution is "
                                         "analytical without iterative or numeric solvers with potentially infinite "
                                         "loops.</ERROR>")
                                else:
                                    s = f"<ERROR>{error}</ERROR>"
                            elif joints is None:
                                s = f"<ERROR>\nReturned no joints - expected {expected}.\n</ERROR>"
                            elif len(joints) != expected:
                                s = (f"<ERROR>\nReturned the wrong number of joints - expected {expected} but got "
                                     f"{len(joints)}.\n</ERROR>")
                            else:
                                # Test the result otherwise and format it.
                                positions, orientations = self.robot.forward_kinematics(lower, upper, joints)
                                reached_position = positions[-1]
                                reached_orientation = orientations[-1]
                                d_p = difference_distance(target_position, reached_position)
                                d_a = difference_angle(target_orientation, reached_orientation) if orientation else 0
                                p = "Successfully reached the target." if reached(d_p, d_a) else ("Failed to reach the "
                                                                                                  "target.")
                                headers = ["Link", "Position", "Orientation"]
                                data = []
                                chain = self.robot.chains[lower][upper]
                                num = len(chain.links)
                                for j in range(num):
                                    data.append([chain.links[j].name, neat(positions[j]), neat(orientations[j])])
                                s = (f"<TEST_SOLUTION>\n{p}\n{tabulate(data, headers, tablefmt='presto')}\n"
                                     "</TEST_SOLUTION>")
                                logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | "
                                             f"{solving} | {mode} | Handle Interactions | Solution tested.")
                        except Exception as e:
                            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | "
                                         f"{mode} | Handle Interactions | Test solution did not respond with valid"
                                         f" parameters: {e}")
                            s = ("<ERROR>\nCould not parse parameters to test the solution; ensure they are all floats."
                                 "</ERROR>")
                    os.makedirs(root, exist_ok=True)
                    with open(os.path.join(root, f"{total}-{MESSAGE_TEST}.txt"), "w", encoding="utf-8",
                              errors="ignore") as file:
                        file.write(s)
                    history.append({"Type": MESSAGE_TEST, "Message": s})
                    return history
            # Otherwise, indicate there was an invalid response.
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Handle "
                         "Interactions | No Python code or functions found; creating message indicating this.")
            s = ("<ERROR>\nYou did not respond with valid code to solve the inverse kinematics or a valid command. "
                 'Ensure code is contained within a Python code block like:\n```python\nprint("Hello world!")\n```\n'
                 "</ERROR>")
            os.makedirs(root, exist_ok=True)
            with open(os.path.join(root, f"{total}-{MESSAGE_FEEDBACK}.txt"), "w", encoding="utf-8",
                      errors="ignore") as file:
                file.write(s)
            history.append({"Type": MESSAGE_FEEDBACK, "Message": s})
            return history
        # Ensure every code does not have any extra running or testing code.
        for i in range(total_codes):
            # Delete any "main" portion.
            codes[i] = codes[i].split('if __name__ == "__main__":')[0].split("if __name__ == '__main__'")[0].strip()
            # Process out only what we need.
            codes[i] = process_code(codes[i]).strip()
        # Use the largest code block if multiple.
        codes.sort(key=len, reverse=True)
        code = codes[0]
        logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Handle "
                     "Interactions | Extracted code.")
        # Save the code so it can be loaded by the program.
        os.makedirs(self.solutions, exist_ok=True)
        with open(code_path, "w", encoding="utf-8", errors="ignore") as file:
            file.write(code)
        # Evaluate the code.
        self.evaluate(lower, upper, orientation, mode)
        s = self.prepare_feedback(lower, upper, orientation, mode)
        # If the code performed perfectly, we are done.
        if s == "":
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Handle "
                         "Interactions | Performed perfectly on the training set; done.")
            os.makedirs(root, exist_ok=True)
            with open(os.path.join(root, f"{total}-{MESSAGE_DONE}.txt"), "w", encoding="utf-8",
                      errors="ignore") as file:
                file.write("Code performed perfectly; interactions with the model are done.")
            return None
        # If there were errors but the maximum number of feedbacks have been given, stop.
        if are_done:
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Code "
                         f"had errors but {MAX_PROMPTS} feedback{' has' if MAX_PROMPTS == 1 else 's have'} been used; "
                         "stopping.")
            os.makedirs(root, exist_ok=True)
            with open(os.path.join(root, f"{total}-{MESSAGE_DONE}.txt"), "w", encoding="utf-8",
                      errors="ignore") as file:
                file.write(f"Code had errors but {MAX_PROMPTS} feedback{' has' if MAX_PROMPTS == 1 else 's have'} been "
                           "used; stopping.")
            return None
        # Otherwise, prepare feedback to provide to the LLM.
        path = os.path.join(root, f"{total}-{MESSAGE_FEEDBACK}.txt")
        os.makedirs(root, exist_ok=True)
        with open(path, "w", encoding="utf-8", errors="ignore") as file:
            file.write(s)
        logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | {solving} | {mode} | Handle "
                     f"Interactions | New feedback saved.")
        history.append({"Type": MESSAGE_FEEDBACK, "Message": s})
        return history

    def __str__(self) -> str:
        """
        Print as a string.
        :return: The details of this solver.
        """
        # If the solver is not valid, return the name and state this.
        if not self.is_valid():
            return f"{self.model} | Invalid"
        is_api = self.url != ""
        # Chat models don't have much to display.
        s = f"{self.robot.name} | {self.model} | "
        if not is_api:
            return f"{s}Chat | Reasoning = {self.reasoning}"
        s += (f"API | Reasoning = {self.reasoning} | Input = "
              f"{'None' if self.input_cost < 0 else f'${self.input_cost}'} | Output = "
              f"{'None' if self.output_cost < 0 else f'${self.output_cost}'} | URL = {self.url} | Inherited = ")
        # Add the names of all inherited models.
        names = []
        for inherited in self.options:
            if self == inherited or inherited.model == self.model:
                continue
            names.append(inherited.model)
        if len(names) == 0:
            s += "None"
        else:
            s += ", ".join(names)
        return s

    def load_code(self, lower: int = 0, upper: int = -1, orientation: bool = False, mode: str = NORMAL,
                  suppress: bool = False) -> (bool, str or None):
        """
        Load the code for a solver.
        :param lower: The starting joint.
        :param upper: The ending joint.
        :param orientation: If this data cares about the orientation or not.
        :param mode: The mode by which the code was achieved.
        :param suppress: If the error for the code not existing should be suppressed.
        :return: If the code was loaded or not along with a reason why it was not loaded.
        """
        # Nothing to do if the solver is not valid.
        if not self.is_valid():
            logging.error(f"{self.model} | Load Code | Solver is not valid.")
            return False, None
        # Ensure valid values.
        lower, upper = self.robot.validate_lower_upper(lower, upper)
        if mode not in [NORMAL, EXTEND, DYNAMIC, CUMULATIVE, TRANSFER]:
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
            return False, None
        # Try to load the inverse kinematics method from the Python file.
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # If the method is not in the file, return.
            if not hasattr(module, "inverse_kinematics"):
                logging.info(f"{self.model} | {lower + 1} to {upper + 1} | Load Code | {solving} | {mode} | Solver "
                             f"'{path}' does not have the method 'inverse_kinematics'.")
                return False, "Solver does not have the method 'inverse_kinematics'."
            method = getattr(module, "inverse_kinematics")
        except Exception as e:
            logging.info(f"{self.model} | {lower + 1} to {upper + 1} | Load Code | {solving} | {mode} | Failed to load "
                         f"'{path}': {e}")
            return False, f"Failed to load the solver: {e}"
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
        return True, None

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
        # Ensure valid values.
        lower, upper = self.robot.validate_lower_upper(lower, upper)
        # See if there is valid code.
        if lower == upper and (orientation or mode != NORMAL):
            return None, 0, None
        # See if the code can be loaded.
        loaded, message = self.load_code(lower, upper, orientation, mode, True)
        if not loaded:
            return None, 0, message
        # Ensure a position.
        solving = TRANSFORM if orientation else POSITION
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
                joints = func_timeout(MAX_TIME, self.code[lower][upper][solving][mode], args=[position])
                elapsed = time.perf_counter() - start_time
            except FunctionTimedOut:
                elapsed = np.inf
                message = ERROR_TIMED_OUT
                logging.info(f"{self.model} | {lower + 1} to {upper + 1} | Run Code | {solving} | {mode} | Error: Code "
                             "timed out.")
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                message = traceback.format_exc()
                logging.info(f"{self.model} | {lower + 1} to {upper + 1} | Run Code | {solving} | {mode} | Error: {e}")
        else:
            orientation = tuple(orientation)
            start_time = time.perf_counter()
            try:
                joints = func_timeout(MAX_TIME, self.code[lower][upper][solving][mode], args=(position, orientation))
                elapsed = time.perf_counter() - start_time
            except FunctionTimedOut:
                elapsed = np.inf
                message = ERROR_TIMED_OUT
                logging.info(f"{self.model} | {lower + 1} to {upper + 1} | Run Code | {solving} | {mode} | Error: Code "
                             f"timed out.")
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                message = traceback.format_exc()
                logging.info(f"{self.model} | {lower + 1} to {upper + 1} | Run Code | {solving} | {mode} | Error: {e}")
        # Parse the joints.
        if joints is not None:
            # If a single float was returned (as should be for single-link chains), make it a list.
            if isinstance(joints, numbers.Number):
                joints = [joints]
            # Otherwise, get the list.
            else:
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

    def evaluate(self, lower: int = 0, upper: int = -1, orientation: bool = False,
                 mode: str = NORMAL) -> None:
        """
        Save the evaluations of this solver.
        :param lower: The starting joint.
        :param upper: The ending joint.
        :param orientation: If we want to solve for orientation.
        :param mode: The solving mode to use.
        :return: Nothing.
        """
        # Nothing to do if the solver is not valid.
        if not self.is_valid():
            logging.error(f"{self.model} | Evaluate | Solver is not valid.")
            return None
        # Ensure valid values.
        lower, upper = self.robot.validate_lower_upper(lower, upper)
        if lower == upper:
            orientation = False
        solving = TRANSFORM if orientation else POSITION
        if mode not in [NORMAL, EXTEND, DYNAMIC, CUMULATIVE, TRANSFER]:
            logging.warning(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Evaluate | Mode "
                            f"'{mode}' not valid, using '{NORMAL}' instead.")
            mode = NORMAL
        # Get the data to run the code against.
        data = self.robot.get_data(lower, upper, False, orientation)
        # If there is no data, there is nothing to evaluate.
        total = len(data)
        if total < 1:
            logging.error(f"{self.model} | {lower + 1} to {upper + 1} | {solving} | {mode} | Evaluate | No data.")
            return None
        # Store results.
        successes = 0
        errors = 0
        total_distance = 0
        total_angle = 0
        total_time = 0
        # The expected number of joints to be returned.
        number = upper - lower + 1
        for point in data:
            # Determine what to test against.
            target_position = point["Position"]
            target_orientation = point["Orientation"] if orientation else None
            # Run the code.
            joints, generation_time, error = self.run_code(lower, upper, mode, target_position, target_orientation)
            total_time += generation_time
            # Store if there was an error.
            if error is not None or joints is None or len(joints) != number:
                # Stop due to any timeout, as these are clearly infinite loops.
                if error == ERROR_TIMED_OUT:
                    successes = 0
                    errors = total
                    total_time = np.inf
                    total_distance = 0
                    total_angle = 0
                    break
                errors += 1
                continue
            # Ensure all returned joints are valid.
            valid = True
            for joint in joints:
                # noinspection PyTypeChecker
                if isinstance(joint, numbers.Number) and not math.isnan(joint):
                    continue
                valid = False
                break
            if not valid:
                errors += 1
                continue
            # See if the move was successful.
            positions, orientations = self.robot.forward_kinematics(lower, upper, joints)
            distance = difference_distance(target_position, positions[-1])
            angle = difference_angle(target_orientation, orientations[-1]) if orientation else 0
            # See if there was an error reaching, which at this point should not happen.
            if math.isnan(distance) or math.isnan(angle):
                errors += 1
                continue
            # If successful, update it.
            if reached(distance, angle):
                successes += 1
                continue
            # Otherwise, add to the failure offsets.
            total_distance += distance
            total_angle += angle
        # Tabulate final results.
        failures = total - successes
        if failures > 0:
            total_distance /= failures
            total_angle /= failures
        successes = neat(successes / total * 100)
        failures = neat(failures / total * 100)
        errors = neat(errors / total * 100)
        total_distance = neat(total_distance)
        total_angle = neat(total_angle)
        total_time = neat(total_time / total)
        # Get all stats
        cost, feedbacks, forwards, testings = self.get_cost(lower, upper, orientation, mode)
        # Get how long it took the LLM to generate the code.
        generation_time = 0
        name = f"{lower}-{upper}-{solving}-{mode}"
        root = os.path.join(self.elapsed, name)
        if os.path.exists(root):
            times = get_files(root)
            for t in times:
                p = os.path.join(root, t)
                with open(p, "r", encoding="utf-8", errors="ignore") as file:
                    s = file.read()
                lines = s.splitlines()
                if len(lines) < 2:
                    logging.error(f"{self.model} | {lower + 1} to {upper + 1} | {solving} | {mode} | Evaluate | "
                                  f"Incorrect elapsed time data in '{p}'.")
                    continue
                try:
                    f = float(lines[1].replace("s", "").strip())
                except Exception as e:
                    logging.error(f"{self.model} | {lower + 1} to {upper + 1} | {solving} | {mode} | Evaluate | Could "
                                  f"not parse time data from '{p}': {e}")
                    continue
                generation_time += f
        # Save the results.
        s = ("Success Rate (%),Failure Rate (%),Error Rate (%),Average Failure Distance,Average Failure Angle (°),"
             "Average Elapsed Time (s),Generation Time (s),Mode,Feedbacks Given,Forwards Kinematics Calls,Testing Calls"
             f",Reasoning,Functions,API,Cost ($)\n{successes}%,{failures}%,{errors}%,{total_distance},"
             f"{total_angle if orientation else 0}°,{total_time} s,{generation_time} s,{mode},{feedbacks},{forwards},"
             f"{testings},{self.reasoning},{self.methods},{self.url != ''},${neat(cost)}")
        os.makedirs(self.results, exist_ok=True)
        with open(os.path.join(self.results, f"{name}.csv"), "w", encoding="utf-8", errors="ignore") as file:
            file.write(s)

    def code_successful(self, lower: int = 0, upper: int = -1, orientation: bool = False,
                        mode: str = NORMAL) -> bool:
        """
        See if a code is completely successful on the training data.
        :param lower: The starting joint.
        :param upper: The ending joint.
        :param orientation: If we want to solve for orientation.
        :param mode: The solving mode to use.
        :return: True if it passes all test cases, false otherwise.
        """
        # Nothing to do if the solver is not valid.
        if not self.is_valid():
            logging.error(f"{self.model} | Code Successful | Solver is not valid.")
            return False
        # Ensure valid values.
        lower, upper = self.robot.validate_lower_upper(lower, upper)
        solving = TRANSFORM if orientation else POSITION
        if mode not in [NORMAL, EXTEND, DYNAMIC, CUMULATIVE, TRANSFER]:
            logging.warning(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Code Successful | Mode "
                            f"'{mode}' not valid, using '{NORMAL}' instead.")
            mode = NORMAL
        # Get the data to run the code against.
        data = self.robot.get_data(lower, upper, True, orientation)
        # If there is no data, there is nothing to give feedback on.
        if len(data) < 1:
            logging.error(f"{self.model} | {lower + 1} to {upper + 1} | {solving} | {mode}| Code Successful | No "
                          f"data.")
            return False
        # The expected number of joints to be returned.
        number = upper - lower + 1
        # Test every data point, stopping if one fails.
        for point in data:
            # Determine what to test against.
            target_position = point["Position"]
            target_orientation = point["Orientation"] if orientation else None
            # Run the code.
            joints, elapsed, error = self.run_code(lower, upper, mode, target_position, target_orientation)
            # If any return values are errors, stop.
            if joints is None or len(joints) != number or error is not None:
                return False
            # See if we reached the target.
            positions, orientations = self.robot.forward_kinematics(lower, upper, joints)
            distance = difference_distance(target_position, positions[-1])
            angle = difference_angle(target_orientation, orientations[-1]) if orientation else 0
            # If we did not, this was a failure so stop.
            if not reached(distance, angle):
                return False
        return True

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
        solving = TRANSFORM if orientation else POSITION
        if mode not in [NORMAL, EXTEND, DYNAMIC, CUMULATIVE, TRANSFER]:
            logging.warning(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare Feedback | Mode "
                            f"'{mode}' not valid, using '{NORMAL}' instead.")
            mode = NORMAL
        # Get the data to run the code against.
        data = self.robot.get_data(lower, upper, True, orientation)
        # If there is no data, there is nothing to give feedback on.
        if len(data) < 1:
            logging.error(f"{self.model} | {lower + 1} to {upper + 1} | {solving} | {mode} | Prepare Feedback | No "
                          f"data.")
            return ""
        # Store what to respond with.
        errors = []
        failures = []
        # The expected number of joints to be returned.
        number = upper - lower + 1
        for point in data:
            # Determine what to test against.
            target_position = point["Position"]
            target_orientation = point["Orientation"] if orientation else None
            # Run the code.
            joints, elapsed, error = self.run_code(lower, upper, mode, target_position, target_orientation)
            # If we got an error, save it.
            if error is not None:
                # Stop immediately if there was a timeout.
                if error == ERROR_TIMED_OUT:
                    errors = ["Code timed out after taking too long. Ensure your solution is analytical without "
                              "iterative or numeric solvers with potentially infinite loops."]
                    break
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
            angle = difference_angle(target_orientation, orientations[-1]) if orientation else 0
            # If we did, this was a success so continue.
            if reached(distance, angle):
                continue
            # Otherwise, detail what the issue was.
            a = f" and orientation {neat(target_orientation)}" if orientation else ""
            b = f" and orientation {neat(orientations[-1])}" if orientation else ""
            failures.append(f"Failed to reach position {neat(target_position)}{a}. Instead reached position "
                            f"{neat(positions[-1])}{b}. The correct joint value"
                            f"{'s were' if len(point['Joints']) > 1 else ' was'} {neat(point['Joints'])} and the "
                            f"joint{'' if len(joints) == 1 else 's'} produced by the code "
                            f"{'was' if len(joints) == 1 else 'were'} {neat(joints)}.")
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

    def prepare_llm(self, lower: int = 0, upper: int = -1, orientation: bool = False, mode: str = NORMAL) -> str:
        """
        Prepare an initial prompt for the LLM.
        :param lower: The starting joint.
        :param upper: The ending joint.
        :param orientation: If we want to solve for orientation.
        :param mode: The solving mode to use.
        :return: The initial prompt for the LLM.
        """
        # Nothing to do if the solver is not valid.
        if not self.is_valid():
            logging.error(f"{self.model} | Prepare LLM | Solver is not valid.")
            return ""
        # Ensure valid values.
        lower, upper = self.robot.validate_lower_upper(lower, upper)
        if mode not in [NORMAL, EXTEND, DYNAMIC, CUMULATIVE, TRANSFER]:
            logging.warning(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | Mode "
                            f"'{mode}' not valid, using '{NORMAL}' instead.")
            mode = NORMAL
        # Cannot do orientation if just a single joint, and we can only run in the normal mode.
        if lower == upper and (orientation or mode != NORMAL):
            return ""
        # No prompt to make if we cannot transfer.
        if not orientation and mode == TRANSFER:
            return ""
        # If there is a better option, do point in performing this.
        best, best_mode, best_cost = self.get_best(lower, upper, orientation)
        if best is not None:
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | "
                         "A cheaper solution is already successful; not doing another prompt.")
            return ""
        # Explain how to use functions.
        mid = "" if self.methods else 'in the "FUNCTIONS" section '
        pre = (" You may respond by either completing the inverse kinematics method or calling either of the two "
               f"provided functions {mid}to help you develop your solution. If you call a function, you will be "
               "provided another response and chance to complete the inverse kinematics method. You may only call a "
               "single function per response.")
        # If this is not a reasoning model, let us manually add in a prompt to do some reasoning.
        if not self.reasoning:
            pre += " Think step by step and show all your work."
        # If the solver using API-based methods, then this is handled in the API formatting.
        if self.methods:
            post = ""
        # Otherwise, explain how to use the commands directly in the prompt.
        else:
            d = ("Test the forward kinematics of the robot, returning the position and orientation of all links in "
                 "world space after setting the joint value")
            if lower == upper:
                j = " value"
                d += ' where "value" is the joint value as a float.'
            else:
                j = ""
                for i in range(lower, upper + 1):
                    j += f" joint{i + 1}"
                if lower + 1 == upper:
                    f's where "joint1" and "joint2" are the joint values as floats.'
                else:
                    d += 's where "joint1"'
                    for i in range(lower + 1, upper):
                        d += f', "joint{i + 1}"'
                    d += f', and "joint{upper}" are the joint values as floats.'
            t = ("Returns the position and orientation of all links in world space after testing your current inverse "
                 'kinematics solution code where "positionX", "positionY", and "positionZ" are the target position')
            p = "TEST_SOLUTION positionX positionY positionZ"
            if orientation:
                t += ', and "orientationX", "orientationY", and "orientationZ" are the target orientation as radians.'
                p += " orientationX orientationY orientationZ"
            else:
                t += "."
            post = ('\n<FUNCTIONS>\n\t<USAGE>\n\tTo use a function, response with the format denoted in the "FORMAT" '
                    "section of the function.\n\t</USAGE>\n\t<FORWARD_KINEMATICS>\n\t\t<FORMAT>\n\t\tFORWARD_KINEMATICS"
                    f"{j}\n\t\t</FORMAT>\n\t\t<DESCRIPTION>\n\t\t{d}\n\t\t</DESCRIPTION>\n\t</FORWARD_KINEMATICS>\n\t"
                    f"<TEST_SOLUTION>\n\t\t<FORMAT>\n\t\t{p}\n\t\t</FORMAT>\n\t\t<DESCRIPTION>\n\t\t{t}\n\t\t"
                    "</DESCRIPTION>\n\t</TEST_SOLUTION>\n</FUNCTIONS>")
        # Perform normal prompts.
        if mode == NORMAL:
            prompt = self.robot.prepare_llm(lower, upper, orientation, pre)
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | Normal prompt "
                         f"prepared.")
            return prompt + post
        # Transfer prompting mode.
        if mode == TRANSFER:
            # See if there is an orientation to model to transfer.
            best, best_mode, cost = self.get_best(lower, upper, False)
            if best is None:
                logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | No chain "
                             "to transfer.")
                return ""
            path = os.path.join(best.solutions, f"{lower}-{upper}-{POSITION}-{best_mode}.py")
            if not os.path.exists(path):
                logging.error(f"{best.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | Best "
                              f"chain to transfer does not exist at '{path}'.")
                return ""
            # Save the inherited information.
            s = f"{best.model}|{lower}|{upper}|{POSITION}|{best_mode}"
            inherited_path = os.path.join(self.interactions, f"{lower}-{upper}-{TRANSFORM}-{TRANSFER}")
            os.makedirs(inherited_path, exist_ok=True)
            inherited_path = os.path.join(inherited_path, f"{INHERITED}.txt")
            with open(inherited_path, "w", encoding="utf-8", errors="ignore") as file:
                file.write(s)
            # Add the transfer prompt portion.
            additional = (f" To help you, a solution for solving the chain for position only is provided in the "
                          '"EXISTING" section. You can use this solution as a starting point to solve for the position '
                          f"and orientation.{pre}")
            prompt = self.robot.prepare_llm(lower, upper, orientation, additional)
            prompt += "\n<EXISTING>\n"
            with open(path, "r", encoding="utf-8", errors="ignore") as file:
                prompt += file.read().strip()
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | Transfer "
                         "prompt prepared.")
            return f"{prompt}\n</EXISTING>{post}"
        # Extending prompting mode.
        if mode == EXTEND:
            # We can only extend a successful lower chain.
            previous = upper - 1
            previous_orientation = orientation and lower != previous
            best, previous_mode, cost = self.get_best(lower, previous, previous_orientation)
            if best is None:
                logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | No chain "
                             "to extend.")
                return ""
            previous_solving = TRANSFORM if previous_orientation else POSITION
            path = os.path.join(best.solutions, f"{lower}-{previous}-{previous_solving}-{previous_mode}.py")
            if not os.path.exists(path):
                logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | Best "
                              f"chain to extend does not exist at '{path}'.")
                return ""
            # Save the inherited information.
            s = f"{best.model}|{lower}|{previous}|{previous_solving}|{previous_mode}"
            inherited_path = os.path.join(self.interactions,
                                          f"{lower}-{upper}-{TRANSFORM if orientation else POSITION}-{EXTEND}")
            os.makedirs(inherited_path, exist_ok=True)
            inherited_path = os.path.join(inherited_path, f"{INHERITED}.txt")
            with open(inherited_path, "w", encoding="utf-8", errors="ignore") as file:
                file.write(s)
            # Add the extending prompt portions.
            total = upper - lower
            plural = "s" if total > 1 else ""
            additional = (f" To help you, a solution for solving the sub-chain of the first {total} link{plural} is "
                          'provided in the "EXISTING" section. This code solved the sub-chain assuming link '
                          f"{total + 1} was the position{' and orientation' if orientation else ''} being solved for. "
                          f"You can use this solution as a starting point to extend for the entire chain.{pre}")
            prompt = self.robot.prepare_llm(lower, upper, orientation, additional)
            prompt += "\n<EXISTING>\n"
            with open(path, "r", encoding="utf-8", errors="ignore") as file:
                prompt += file.read().strip()
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | Extended "
                         "prompt prepared.")
            return f"{prompt}\n</EXISTING>{post}"
        # Handle dynamic mode.
        if mode == DYNAMIC:
            # Get the best possible dynamic option.
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | Beginning best"
                         " dynamic chain search.")
            best = self.get_dynamic(lower, upper, orientation)
            if best is None:
                logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | Not "
                             "performing a dynamic prompt as no options.")
                return ""
            total = len(best)
            # If somehow nothing was returned, this is an error.
            if total == 0:
                logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | Empty "
                              "dynamic chain returned.")
                return ""
            # If this is just the same as a normal prompt, lets not waste resources running it.
            if total == 1:
                logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | Not "
                             "performing a dynamic prompt as just a single chain.")
                return ""
            # Load the codes.
            codes = []
            inherit = ""
            for chain in best:
                solver = chain["Solver"]
                c_lower = chain["Lower"]
                c_upper = chain["Upper"]
                c_solving = chain["Solving"]
                c_mode = chain["Mode"]
                path = os.path.join(solver.solutions,
                                    f"{c_lower}-{c_upper}-{c_solving}-{c_mode}.py")
                if not os.path.exists(path):
                    logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | Part "
                                  f"of dynamic chain at '{path}' does not exist.")
                    return ""
                with open(path, "r", encoding="utf-8", errors="ignore") as file:
                    codes.append(file.read().strip())
                # Add the inherited data.
                t = f"{solver.model}|{c_lower}|{c_upper}|{c_solving}|{c_mode}"
                if inherit == "":
                    inherit = t
                else:
                    inherit += f"\n{t}"
            # Save the inherited data.
            path = os.path.join(self.interactions, f"{lower}-{upper}-{TRANSFORM if orientation else POSITION}-{mode}")
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, f"{INHERITED}.txt")
            with open(path, "w", encoding="utf-8", errors="ignore") as file:
                file.write(inherit)
            # Explain the dynamic chains.
            additional = (' To help you, solutions for sub-chains have been provided in the "EXISTING" sections. Each '
                          "code solved a sub-link assuming their last link was the position"
                          f"{' and orientation' if orientation else ''} being solved for. You can use these solutions "
                          "as a starting point to extend for the entire chain.")
            # State what sub-chain each dynamic code is for.
            for i in range(total):
                c_lower = best[i]["Lower"] - lower
                c_upper = best[i]["Upper"] - lower
                ending = f"joint {c_lower + 1}" if c_lower == c_upper else f"joints {c_lower + 1} to {c_upper + 1}"
                additional += f" Existing code {i + 1} solved {ending}."
            # Build the prompt.
            prompt = self.robot.prepare_llm(lower, upper, orientation, additional + pre)
            # Add the existing codes to the prompt.
            for i in range(total):
                prompt += f"\n<EXISTING {i + 1}>\n{codes[i]}\n</EXISTING {i + 1}>"
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | Dynamic prompt"
                         " prepared.")
            return f"{prompt}{post}"
        if lower == upper - 1:
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | Not doing a"
                         " cumulative prompt as this is just 2 DOF which would mean either a dynamic or extending.")
            return ""
        # Lastly, do cumulative prompts, collecting all solutions to sub-chains.
        sequences = []
        inherited = ""
        sequence_upper = upper + 1
        for sub_lower in range(lower, sequence_upper):
            for sub_upper in range(sub_lower, sequence_upper):
                sub_orientation = False if sub_lower == sub_upper else orientation
                best, best_mode, best_cost = self.get_best(sub_lower, sub_upper, sub_orientation)
                if best is None:
                    continue
                sub_solving = TRANSFORM if sub_orientation else POSITION
                path = os.path.join(SOLUTIONS, self.robot.name, best.model,
                                    f"{sub_lower}-{sub_upper}-{sub_solving}-{best_mode}.py")
                if not os.path.exists(path):
                    logging.error(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | "
                                  f"Best cumulative part from {lower + 1} to {upper + 1} does not exist: {path}")
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as file:
                    code = file.read().strip()
                sequences.append({"Lower": sub_lower, "Upper": sub_upper, "Code": code})
                current_inherited = f"{best.model}|{sub_lower}|{sub_upper}|{sub_solving}|{best_mode}"
                if inherited == "":
                    inherited = current_inherited
                else:
                    inherited += f"\n{current_inherited}"
        # If there are no sub-chain solutions, there is nothing to do.
        total = len(sequences)
        if total == 0:
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | Not "
                         "performing a cumulative prompt as there are no parts to prompt with.")
            return ""
        # See if this is just an extending solution.
        if total == 1 and sequences[0]["Lower"] == lower and sequences[0]["Upper"] == upper - 1:
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | Not "
                         "performing a cumulative prompt as this is the same as an extending prompt.")
            return ""
        # See if this is just a dynamic prompt.
        covered = []
        for i in range(lower, upper):
            covered.append(0)
        has_overlap = False
        for sequence in sequences:
            sequence_lower = lower - sequence["Lower"]
            sequence_upper = upper - sequence["Upper"]
            for i in range(sequence_lower, sequence_upper):
                covered[i] += 1
                if covered[i] > 1:
                    has_overlap = True
        for c in covered:
            if c == 0:
                logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | Not "
                             "performing a cumulative prompt as not all links are covered by a solution.")
                return ""
        if not has_overlap:
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | Not "
                         "performing a cumulative prompt as this is the same as a dynamic prompt.")
            return ""
        # Actually build the cumulative prompt.
        additional = (' To help you, solutions for sub-chains have been provided in the "EXISTING" sections. Each '
                      "code solved a sub-link assuming their last link was the target being solved for."
                      f" You can use these solutions as a starting point to extend for the entire chain.")
        # State what sub-chain each dynamic code is for.
        for i in range(total):
            c_lower = sequences[i]["Lower"] - lower
            c_upper = sequences[i]["Upper"] - lower
            ending = f"joint {c_lower + 1}" if c_lower == c_upper else f"joints {c_lower + 1} to {c_upper + 1}"
            additional += f" Existing code {i + 1} solved {ending}."
        # Build the prompt.
        prompt = self.robot.prepare_llm(lower, upper, orientation, additional + pre)
        # Add the existing codes to the prompt.
        for i in range(total):
            prompt += f"\n<EXISTING {i + 1}>\n{sequences[i]['Code']}\n</EXISTING {i + 1}>"
        logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Prepare LLM | Cumulative prompt"
                     " prepared.")
        # Save the inherited data.
        path = os.path.join(self.interactions, f"{lower}-{upper}-{TRANSFORM if orientation else POSITION}-{mode}")
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, f"{INHERITED}.txt")
        with open(path, "w", encoding="utf-8", errors="ignore") as file:
            file.write(inherited)
        return f"{prompt}{post}"

    def get_dynamic(self, lower: int = 0, upper: int = -1,
                    orientation: bool = False) -> list[dict[str, int or str]] or None:
        """
        Get the best dynamic chain.
        :param lower: The starting joint.
        :param upper: The ending joint.
        :param orientation: If we want to solve for orientation.
        :return: The best dynamic chain or none if none were found.
        """
        # Nothing to do if the solver is not valid.
        if not self.is_valid():
            logging.error(f"{self.model} | Get Dynamic | Solver is not valid.")
            return None
        # Ensure valid values.
        lower, upper = self.robot.validate_lower_upper(lower, upper)
        # Cannot do orientation if just a single joint, and we can only run in the normal mode.
        current_orientation = False if lower == upper else orientation
        # See if we already have a full solution here, in which case there is no point in looking for a dynamic one.
        best, best_mode, best_cost = self.get_best(lower, upper, current_orientation)
        if best is not None:
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Get Dynamic | Found a "
                         f"successful solver solving for '{TRANSFORM if current_orientation else POSITION}' in mode "
                         f"'{best_mode}'.")
            return ([{"Solver": best, "Lower": lower, "Upper": upper,
                      "Solving": TRANSFORM if current_orientation else POSITION, "Mode": best_mode}])
        # If this was a base case, there are no valid options.
        if lower == upper:
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Get Dynamic | No successful "
                         "base cases.")
            return None
        # Otherwise, let us try to get the best possible sub-chain and use it.
        best = None
        best_size = 0
        best_bottom = 0
        for split in range(lower, upper):
            # Try to get the bottom dynamic portion.
            bottom = self.get_dynamic(lower, split, orientation)
            if bottom is None:
                logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Get Dynamic | No bottom "
                             f"found from {lower + 1} to {split + 1}.")
                continue
            # Try to get the top dynamic portion.
            top = self.get_dynamic(split + 1, upper, orientation)
            if top is None:
                logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Get Dynamic | No top "
                             f"found from {split + 2} to {upper + 1}.")
                continue
            bottom_size = len(bottom)
            top_size = len(top)
            total_size = bottom_size + top_size
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Get Dynamic | Chains found "
                         f"from {lower + 1} to {upper + 1} and {split + 2} to {upper + 1}. Size is {total_size} "
                         f"({bottom_size} + {top_size}).")
            # Check if this is a better dynamic chain which has been found.
            if best is not None:
                # If the existing number of sub-chains is less, don't do anything.
                if best_size <= total_size:
                    logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Get Dynamic | Best "
                                 f"size of {best_size} is better than the new solution of {total_size} "
                                 f"({bottom_size} + {top_size}).")
                    continue
                # If the sizes are the same, only keep the new chain if the bottom size is larger.
                if best_size == total_size and best_bottom < bottom_size:
                    logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Get Dynamic | New "
                                 f"and best solution have the same size of {best_size} but the best has a larger lower "
                                 f"chain of {best_bottom} compared to the new {bottom_size}.")
                    continue
                logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Get Dynamic | New and "
                             f"best solution have the same size of {best_size} but the new has a larger lower chain of "
                             f"{bottom_size} compared to the current of {best_bottom}.")
            else:
                logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Get Dynamic | First "
                             "successful chain found.")
            # Save the new best dynamic chain.
            best_size = total_size
            best_bottom = bottom_size
            best = []
            for chain in bottom:
                best.append(chain)
            for chain in top:
                best.append(chain)
        if best is None:
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Get Dynamic | No successful "
                         "option found.")
        else:
            logging.info(f"{self.model} | {self.robot.name} | {lower + 1} to {upper + 1} | Get Dynamic | Successful "
                         "option found.")
        return best

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
        c = len(value)
        s = "["
        for i in range(c):
            if i == 0:
                s += neat(value[i])
            else:
                s += f", {neat(value[i])}"
        return f"{s}]"
    # Otherwise, clean the value.
    formatted = str(value)
    # Ensure this is not in exponent form.
    if "e" in formatted:
        # If it is in exponent form, we need to expand it as a decimal value.
        try:
            formatted = str(Decimal(value))
        except Exception as e:
            logging.error(f"Could not neatly format '{value}' as a decimal: {e}")
    # Strip any extra zeros from the decimals, removing the decimal space if nothing remains.
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    # Failsafe check to ensure we actually return a number which should never be needed.
    return "0" if formatted == "" else formatted


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


def get_directories(directory) -> list[str]:
    """
    Get all directories in a directory.
    :param directory: The directory to get the directories from.
    :return: The names of all directories in the directory.
    """
    return [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]


def llm_ik(robots: str or list[str] or None = None, max_length: int = 0, orientation: bool = True,
           types: str = TRANSFER, feedbacks: int = MAX_PROMPTS, examples: int = EXAMPLES, training: int = TRAINING,
           evaluating: int = EVALUATING, seed: int = SEED, distance_error: float = DISTANCE_ERROR,
           angle_error: float = ANGLE_ERROR, run: bool = False, cwd: str or None = None, level: str = "INFO",
           bypass: bool = False, wait: int = WAIT) -> None:
    """
    Run LLM inverse kinematics.
    :param robots: The names of the robots.
    :param max_length: The maximum chain length to run.
    :param orientation: If we want to solve for orientation in addition to position.
    :param types: The highest solving type to run.
    :param feedbacks: The max number of times to give feedback.
    :param examples: The number of examples to give with feedbacks.
    :param training: The number of training samples.
    :param evaluating: The number of evaluating samples.
    :param seed: The samples generation seed.
    :param distance_error: The acceptable distance error.
    :param angle_error: The acceptable angle error.
    :param run: Enable API running.
    :param cwd: The working directory.
    :param level: The logging level.
    :param bypass: Bypass the confirmation for API running.
    :param wait: How long to wait between API calls.
    :return: Nothing.
    """
    # Set the logging level.
    level = level.strip().upper()
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
    else:
        logging.info(f"Set '{cwd}' as the working directory.")
    # Set all paths relative to the working directory and make sure the "required" folders for a human to use exist.
    global ROBOTS
    global MODELS
    global PROVIDERS
    global KEYS
    global INFO
    global INTERACTIONS
    global ELAPSED
    global SOLUTIONS
    global RESULTS
    global TOKENS
    ROBOTS = os.path.join(cwd, ROBOTS)
    MODELS = os.path.join(cwd, MODELS)
    PROVIDERS = os.path.join(cwd, PROVIDERS)
    INTERACTIONS = os.path.join(cwd, INTERACTIONS)
    ELAPSED = os.path.join(cwd, ELAPSED)
    SOLUTIONS = os.path.join(cwd, SOLUTIONS)
    RESULTS = os.path.join(cwd, RESULTS)
    INFO = os.path.join(cwd, INFO)
    KEYS = os.path.join(cwd, KEYS)
    TOKENS = os.path.join(cwd, TOKENS)
    os.makedirs(ROBOTS, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(PROVIDERS, exist_ok=True)
    os.makedirs(KEYS, exist_ok=True)
    # Get the solving types.
    if types not in [NORMAL, EXTEND, DYNAMIC, CUMULATIVE, TRANSFER]:
        logging.warning(f"Solving mode '{types}' not valid; using '{NORMAL}'.")
        types = NORMAL
    else:
        logging.info(f"Solving up to mode '{types}'.")
    # Get the orientation types we wish to solve for.
    if orientation:
        logging.info("Solving for both position and transform.")
    else:
        logging.info(f"Solving for only position only.")
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
    global MAX_PROMPTS
    MAX_PROMPTS = feedbacks
    logging.info(f"Providing {MAX_PROMPTS} feedback{'' if MAX_PROMPTS == 1 else 's'}.")
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
    if max_length < 1:
        logging.info("Computing chains of all lengths.")
    else:
        logging.info(f"Computing on chains up to {max_length} long.")
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
            robots = []
        else:
            robots = [robots]
    # Otherwise, if a list, ensure all passed robots exist.
    else:
        models = []
        for robot in robots:
            if robot not in existing and f"{robot}.urdf" not in existing:
                logging.warning(f"Robot '{robot}' does not exist in '{ROBOTS}'; removing it.")
                continue
            models.append(robot)
        if len(models) < 1:
            logging.error(f"No valid robots were passed.")
        robots = models
    # Load all robots.
    created_robots = []
    for name in existing:
        robot = Robot(name)
        if robot.is_valid():
            created_robots.append(robot)
    if len(created_robots) < 1:
        logging.error("No robots could be successfully loaded; nothing to perform on.")
        return None
    total = len(created_robots)
    logging.info(f"{total} robot{'s' if total > 1 else ''} loaded.")
    # Get the robots we will actually perform API calls on.
    perform = []
    for robot in robots:
        name = robot.replace(".urdf", "")
        for c in created_robots:
            if c.name == name:
                perform.append(name)
    robots = perform
    # See which models exist in the core folder.
    models = get_files(MODELS)
    # Clean out only to the models we are interested in.
    total = len(models)
    # If there are no models in the first place, there is nothing to check.
    if total < 1:
        logging.warning("No models; can only perform built-in IKPy inverse kinematics.")
    # Otherwise, ensure our passed models are valid.
    else:
        # Clean the names of models.
        for i in range(total):
            models[i] = models[i].replace(".txt", "")
    # If there is at least one model we should load, let us try to fully load it.
    created_models = []
    if total > 0:
        # Try for every LLM paired with every robot.
        for name in models:
            for robot in created_robots:
                model = Solver(name, robot)
                if model.is_valid():
                    created_models.append(model)
        total = len(created_models)
        if total < 1:
            logging.warning("No models loaded; can only perform built-in IKPy inverse kinematics.")
        else:
            logging.info(f"{total} model{'s' if total > 1 else ''} loaded.")
    # Load robot data.
    for robot in created_robots:
        robot.load_data()
    # Sort API models from cheapest to most expensive.
    created_models = sorted(created_models, key=lambda x: (x.robot.name, x.url is not None, x.reasoning, x.output_cost,
                                                           x.input_cost, x.model))
    # Get the API models.
    api_models = []
    for s in created_models:
        if s.url != "" and s.input_cost >= 0 and s.output_cost >= 0:
            api_models.append(s)
    # Set up inheriting for API models.
    for solver in api_models:
        options = []
        # Get the inheriting options.
        for option in api_models:
            # Check basic matches.
            if solver == option or solver.robot != option.robot:
                continue
            # Reasoning models can inherit all non-reasoning models.
            if solver.reasoning and not option.reasoning:
                options.append(option)
                continue
            # Otherwise, only inherit cheaper models.
            if (solver.input_cost < option.input_cost or solver.output_cost < option.output_cost or
                    (not solver.reasoning and option.reasoning)):
                continue
            options.append(option)
        # No point in calling if there is nothing to inherit.
        if len(options) > 0:
            solver.set_inherited(options)
    # Log the inherited options.
    for solver in created_models:
        logging.info(solver)
    if run:
        total_robots = len(robots)
        total_models = len(api_models)
        if total_robots > 0 and total_models > 0:
            # Unless we bypassed the API call checking, confirm we want to run up to the potential number of API calls.
            if not bypass:
                calls = 0
                total_feedbacks = 1 + MAX_PROMPTS
                total_orientations = 2 if orientation else 1
                if types == TRANSFER:
                    total_types = 5
                elif types == CUMULATIVE:
                    total_types = 4
                elif types == DYNAMIC:
                    total_types = 3
                elif types == EXTEND:
                    total_types = 2
                else:
                    total_types = 1
                # Check every robot which supports API calls.
                for robot in created_robots:
                    if robot.name in robots:
                        # The number of chains is the summation of joints, less the last for the single-mode singles.
                        subs = sum(range(1, max(robot.joints, max_length) - 1))
                        # Every chain can have full feedbacks across solving configurations plus the basic solvers.
                        calls += subs * total_feedbacks * total_orientations * total_types + robot.joints
                # Each will be called by every solver.
                calls *= total_models
                s = (f"Performing API calls on {total_robots} robot{'s' if total_robots > 1 else ''} and {total_models}"
                     f" model{'s' if total_models > 1 else ''} with {MAX_PROMPTS} feedback"
                     f"{'' if feedbacks == 1 else 's'} resulting in up to {calls} LLM API call"
                     f"{'s' if calls > 1 else ''}. Confirm if you accept making up to these potential {calls} LLM API "
                     f"call{'s' if calls > 1 else ''} [y/n]: ")
                response = input(s)
                if not response.upper() == "Y":
                    run = False
                    logging.info(f"Declined running up to {calls} LLM API call{'s' if calls > 1 else ''}; disabling API"
                                 f" calls.")
                else:
                    logging.info(f"Confirmed running up to {calls} LLM API call{'s' if calls > 1 else ''}.")
            else:
                logging.info("Running LLM API calls with confirmation bypassed.")
        else:
            logging.info("Not running LLM API calls as there are no selected robots or LLMs.")
            run = False
    else:
        logging.info("Not running LLM API calls.")
    if run and wait > 1:
        logging.info(f"Waiting for {wait} second {'' if wait == 1 else 's'} between API calls.")
    # Run the solvers, making API calls only on those that should be.
    for solver in created_models:
        if solver.url == "" or run:
            run_instance = solver.robot.name in robots and solver.model in models
            if not solver.perform(orientation, types, max_length, run_instance, wait):
                logging.error("Not performing any more API calls as there were errors.")
                run = False
        # Perform one final evaluation for the solver.
        for lower in range(0, solver.robot.joints):
            for upper in range(lower, solver.robot.joints):
                for o in [False, True]:
                    for m in [NORMAL, EXTEND, DYNAMIC, CUMULATIVE, TRANSFER]:
                        # Only generate for those which produced a code file.
                        if os.path.exists(os.path.join(solver.solutions, f"{lower}-{upper}-"
                                                                         f"{TRANSFORM if o else POSITION}-{m}.py")):
                            solver.evaluate(lower, upper, o, m)
    # Evaluate all robots.
    for robot in created_robots:
        robot.evaluate()


if __name__ == "__main__":
    # Configure the argument parser.
    parser = argparse.ArgumentParser(description="LLM Inverse Kinematics")
    parser.add_argument("-r", "--robots", type=str or list[str] or None, default=None, help="The names of the robots.")
    parser.add_argument("-m", "--max", type=int, default=0, help="The maximum chain length to run.")
    parser.add_argument("-o", "--orientation", type=bool, default=True, help="If we want to solve for orientation "
                                                                             "in addition to position.")
    parser.add_argument("-t", "--types", type=str, default=TRANSFER, help="The highest solving type to run.")
    parser.add_argument("-f", "--feedbacks", type=int, default=MAX_PROMPTS, help="The max number of times to give "
                                                                                 "feedback.")
    parser.add_argument("-e", "--examples", type=int, default=EXAMPLES, help="The number of examples to give with "
                                                                             "feedbacks.")
    parser.add_argument("-a", "--training", type=int, default=TRAINING, help="The number of training samples.")
    parser.add_argument("-v", "--evaluating", type=int, default=EVALUATING, help="The number of evaluating samples.")
    parser.add_argument("-s", "--seed", type=int, default=SEED, help="The samples generation seed.")
    parser.add_argument("-d", "--distance", type=float, default=DISTANCE_ERROR, help="The acceptable distance error.")
    parser.add_argument("-n", "--angle", type=float, default=ANGLE_ERROR, help="The acceptable angle error.")
    parser.add_argument("-c", "--cwd", type=str or None, default=None, help="The working directory.")
    parser.add_argument("-l", "--logging", type=str, default="INFO", help="The logging level.")
    parser.add_argument("-w", "--wait", type=int, default=WAIT, help="How long to wait between API calls.")
    parser.add_argument("-u", "--run", action="store_true", help="Enable API running.")
    parser.add_argument("-b", "--bypass", action="store_true", help="Bypass the confirmation for API running.")
    args = parser.parse_args()
    # Run the program.
    llm_ik(args.robots, args.max, args.orientation, args.types, args.feedbacks, args.examples, args.training,
           args.evaluating, args.seed, args.distance, args.angle, args.run, args.cwd, args.logging, args.bypass,
           args.wait)
