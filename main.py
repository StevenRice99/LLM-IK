import logging
import os.path

import numpy as np
import pytorch_kinematics as pk

# Configure logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


class Robot:
    def __init__(self, name: str, first: int = 0, last: int = -1):
        self.name = os.path.splitext(os.path.basename(name))[0]
        if not os.path.exists(name):
            self.first = -1
            self.last = -1
            self.joints = 0
            self.chain = None
            logging.error(f"{self.name} | File '{name}' does not exist.")
            return
        if not os.path.isfile(name):
            self.first = -1
            self.last = -1
            self.joints = 0
            self.chain = None
            logging.error(f"{self.name} | '{name}' is not a file.")
            return
        try:
            # noinspection PyTypeChecker
            chain = pk.build_chain_from_urdf(open(name, mode="rb").read())
        except Exception as e:
            self.first = -1
            self.last = -1
            self.joints = 0
            self.chain = None
            logging.error(f"{self.name} | Could not load '{name}': {e}")
            return
        joints = chain.get_joint_parent_frame_names(True)
        count = len(joints)
        if last < 0:
            last = count - 1
        if first < 0:
            first = 0
        elif first > last:
            first = last
        self.first = first
        self.last = last
        try:
            self.chain = pk.SerialChain(chain, joints[self.last], joints[self.first])
            self.joints = len(self.chain.get_joint_parent_frame_names(True))
        except Exception as e:
            self.first = -1
            self.last = -1
            self.joints = 0
            self.chain = None
            logging.error(f"{self.name} | Could not build a serial chain from joints {first} to {last}: {e}")
            return
        logging.info(self)

    def __str__(self):
        if self.chain is None:
            return f"{self.name} | No loaded data."
        s = f"{self.name} | {self.joints} Joints from index {self.first} to {self.last}"
        joints = self.chain.get_joint_parent_frame_names(True)
        count = len(joints)
        for i in range(count):
            if i == 0:
                s += f": {joints[i]}"
            else:
                s += f", {joints[i]}"
        return s

    def forward_kinematics(self, joints: list[float]) -> (list[float], list[float]):
        if self.chain is None:
            logging.error(f"{self.name} | Forward Kinematics | No loaded data, cannot perform forward kinematics.")
            return [0, 0, 0], [0, 0, 0, 0]
        values = np.zeros(self.joints)
        passed = len(joints)
        upper = min(passed, self.joints)
        if passed < self.joints:
            logging.warning(f"{self.name} | Forward Kinematics | passed {passed} joints being {joints} when the chain "
                            f"has {self.joints}. Will pad remaining upper joints with zeros.")
        elif passed > self.joints:
            logging.warning(f"{self.name} | Forward Kinematics | passed {passed} joints being {joints} when the chain "
                            f"has {self.joints}. Will only use the first {self.joints} values.")
        for i in range(upper):
            values[i] = joints[i]
        try:
            m = self.chain.forward_kinematics(values, True).get_matrix()
            pos = m[:, :3, 3].flatten().tolist()
            rot = pk.matrix_to_quaternion(m[:, :3, :3]).flatten().tolist()
            logging.info(f"{self.name} | Forward Kinematics of {values} | Position = {pos} | Rotation = {rot}")
            return pos, rot
        except Exception as e:
            logging.error(f"{self.name} | Forward Kinematics of {values} | Error performing: {e}")
            return [0, 0, 0], [0, 0, 0, 0]


if __name__ == '__main__':
    robot = Robot("ur5.urdf", first=0, last=-1)
    robot.forward_kinematics([0, 0, 0, 0, 0, 0])
