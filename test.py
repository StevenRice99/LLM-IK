import os

import mujoco
import mujoco.viewer
import numpy as np

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path(os.path.join(os.getcwd(), "models", "universal_robots_ur5e", "ur5e.xml"))
data = mujoco.MjData(model)

# Initialize the viewer if you want to visualize the simulation
viewer = mujoco.viewer.launch_passive(model, data)

# Set desired joint positions (qpos)
desired_joint_positions = np.array([1, 1, 1])  # Example values

# Ensure the length of desired_joint_positions matches the number of joints you are updating
data.qpos[:len(desired_joint_positions)] = desired_joint_positions

# Optionally, you can also reset joint velocities (qvel) if needed
data.qvel[:len(desired_joint_positions)] = 0

# Perform forward kinematics to apply the changes
mujoco.mj_forward(model, data)

# Run the simulation for a few steps to see the effect
while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.sync()

# Properly close the viewer (if needed)