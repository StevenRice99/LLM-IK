import logging
import os

import mujoco
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class Robot:
    def __init__(self, name: str, limits: bool = False, collisions: bool = False, first: int = 0, last: int = 0):
        self.name = name
        self.limits = limits
        self.positions = []
        self.orientations = []
        self.axes = []
        self.lower = []
        self.upper = []
        self.types = []
        self.joint_names = []
        self.tcp_name = ""
        self.first = first
        self.last = last
        path = os.path.join(os.getcwd(), "Models", self.name, "Model.xml")
        if not os.path.exists(path):
            logging.error(f"Model '{self.name}' not found at '{path}'.")
            self.first = 1
            self.last = 1
            return
        try:
            # The method is meant to be used like this, but the Mujoco API itself is defined wrong.
            # This is to hide that warning which we are otherwise handling properly.
            # noinspection PyArgumentList
            self.model = mujoco.MjModel.from_xml_path(filename=path, assets=None)
        except Exception as e:
            logging.error(f"Model '{self.name}' could not be loaded: {e}.")
            self.first = 1
            self.last = 1
            return
        logging.info(f"Model '{self.name}' loaded from '{path}'.")
        try:
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            logging.error(f"Data '{self.name}' could not be loaded: {e}.")
            self.first = 1
            self.last = 1
            return
        logging.info(f"Data for model '{self.name}' loaded.")
        mujoco.mj_forward(self.model, self.data)
        # How many
        if self.last < 1 or self.last > self.model.nsite:
            self.last = self.model.nsite
        if first < 1:
            self.first = 1
        elif first >= self.last:
            self.first = self.last
        first_id = self.model.site_bodyid[self.first - 1]
        first_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, self.first - 1)
        logging.info(f"First site at index {self.first - 1} ID is {first_id} and name is {first_name}.")
        last_id = self.model.site_bodyid[self.last - 1]
        tcp_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, self.last - 1)
        logging.info(f"Last site (TCP) at index {self.last - 1} ID is {last_id} and name is {tcp_name}.")
        keep = []
        while last_id != 0:
            keep.append(last_id)
            logging.info(f"Body ID {last_id} with name "
                         f"'{mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, last_id)}' being kept.")
            if last_id < first_id:
                break
            last_id = self.model.body_parentid[last_id]
        # No previous offset for the first joint.
        previous_offset = np.array([0, 0, 0])
        # Loop every joint.
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            self.joint_names.append(joint_name)
            body = self.model.jnt_bodyid[i]
            # Not interested in later joints.
            if body not in keep:
                logging.info(f"Joint {i + 1} | {joint_name} | Body ID = {body} | Discarded")
                continue
            # Get the type of joint.
            joint_type = self.model.jnt_type[i]
            if joint_type == 3:
                joint_type = "Revolute"
            elif joint_type == 2:
                joint_type = "Linear"
            elif joint_type == 1:
                joint_type = "Spherical"
            else:
                joint_type = "Free"
            self.types.append(joint_type)
            # Some joints may be offset from their body, so we need to account for that.
            offset = self.model.jnt_pos[i]
            # Get this position taking into account how much the previous joint was offset.
            if len(self.positions) < 1:
                core = np.array([0, 0, 0])
            else:
                core = self.model.body_pos[body]
            self.positions.append(core + offset - previous_offset)
            # Set the offset for the next iteration.
            previous_offset = offset
            self.orientations.append(self.model.body_quat[body])
            self.axes.append(self.model.jnt_axis[i])
            # Get limits.
            if self.model.jnt_limited[i]:
                # If we are not looking to enforce limits, remove them.
                if not limits:
                    self.model.jnt_limited[i] = 0
                    self.model.jnt_range[i][0] = -3.14159
                    self.model.jnt_range[i][1] = 3.14159
                self.lower.append(self.model.jnt_range[i][0])
                self.upper.append(self.model.jnt_range[i][1])
            else:
                self.lower.append(-3.14159)
                self.upper.append(3.14159)
            logging.info(f"Joint {i + 1} | {joint_name} | Body ID = {body} | {joint_type} | Position = "
                         f"{self.positions[-1]} | Orientation = {self.orientations[-1]} | Axes = {self.axes[-1]} | "
                         f"Lower = {self.lower[-1]} | Upper = {self.upper[-1]}")
        # Turn off collisions if needed.
        if not collisions:
            logging.info("Disabling collisions.")
            for i in range(self.model.ngeom):
                self.model.geom_contype[i] = 0
                self.model.geom_conaffinity[i] = 0
                self.model.geom_condim[i] = 1
        # If there are collisions, turn them off for joints we do not care about.
        else:
            logging.info("Ensuring only valid collisions.")
            for i in range(self.model.ngeom):
                body = self.model.geom_bodyid[i]
                if body in keep:
                    continue
                logging.info(f"Disabling collisions for body ID {body}.")
                self.model.geom_contype[i] = 0
                self.model.geom_conaffinity[i] = 0
                self.model.geom_condim[i] = 1
        logging.info(f"Loaded '{name}'.")

    def __str__(self) -> str:
        s = ""
        n_joints = len(self.positions)
        for i in range(n_joints):
            pos = self.positions[i]
            quat = self.orientations[i]
            axis = self.axes[i]
            s += (f"Joint {i + 1} | {self.types[i]} | Position = [{neat(pos[0])}, {neat(pos[1])}, {neat(pos[2])}] | "
                  f"Orientation = [{neat(quat[0])}, {neat(quat[1])}, {neat(quat[2])}, {neat(quat[3])}] | Axes = "
                  f"[{neat(axis[0])}, {neat(axis[1])}, {neat(axis[2])}]")
            if self.limits:
                s += f" | Lower = {neat(self.lower[i])} | Upper = {neat(self.upper[i])}"
            s += "\n"
        return s


def neat(value) -> str:
    """
    Format a numerical value with no trailing zeros.
    :param value: The numerical value.
    :return: The value as a formatted string.
    """
    return f"{value:.8f}".rstrip("0").rstrip(".")


if __name__ == "__main__":
    Robot("Custom", first=1, last=6)
