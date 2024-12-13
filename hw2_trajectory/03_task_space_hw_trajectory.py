"""Task space (operational space) control example.

This example demonstrates how to implement task space control for a robot arm,
allowing it to track desired end-effector positions and orientations. The example
uses a simple PD control law but can be extended to more sophisticated controllers.

Key Concepts Demonstrated:
    - Task space control implementation
    - End-effector pose tracking
    - Real-time target visualization
    - Coordinate frame transformations

Example:
    To run this example:
    
    $ python 03_task_space.py

Notes:
    - The target pose can be modified interactively using the MuJoCo viewer
    - The controller gains may need tuning for different trajectories
    - The example uses a simplified task space controller for demonstration
"""

import numpy as np
from simulator import Simulator
from pathlib import Path
from typing import Dict
import os 
import pinocchio as pin
import matplotlib.pyplot as plt

def task_space_controller(q: np.ndarray, dq: np.ndarray, t: float, desired: Dict) -> np.ndarray:
    """Example task space controller."""
    
    # kp = np.array([10, 10, 10, 10, 10, 10])
    # kd = np.array([3, 3, 3, 2, 2, 2])
    kp = np.array([150, 150, 150, 150, 150, 150])
    kd = np.array([45, 45, 45, 45, 45, 45])
    dq_des = np.array([0, 0, 0, 0, 0, 0])
    ddq_des = np.array([0, 0, 0, 0, 0, 0])

    
    pin.computeAllTerms(model, data, q, dq)
    # Convert desired pose to SE3
    desired_position = desired['pos']
    desired_quaternion = desired['quat'] # [w, x, y, z] in MuJoCo format
    desired_quaternion_pin = np.array([*desired_quaternion[1:], desired_quaternion[0]]) # Convert to [x,y,z,w] for Pinocchio
    # Convert to pose and SE3
    desired_pose = np.concatenate([desired_position, desired_quaternion_pin])
    desired_se3 = pin.XYZQUATToSE3(desired_pose)
    p_des = desired_se3.translation
    R_des = desired_se3.rotation

    # Get end-effector frame
    ee_frame_id = model.getFrameId("end_effector")
    
    J = np.zeros((6,6))
    J_w = pin.getFrameJacobian(model, data, ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
    J_l = pin.getFrameJacobian(model, data, ee_frame_id, pin.LOCAL)
    J[:3,:] = J_w[:3,:]
    J[3:,:] = J_l[3:,:]
    dJ = np.zeros((6,6))
    dJ_w = pin.getFrameJacobianTimeVariation(model, data, ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
    dJ_l = pin.getFrameJacobianTimeVariation(model, data, ee_frame_id, pin.LOCAL)
    dJ[:3,:] = dJ_w[:3,:]
    dJ[3:,:] = dJ_l[3:,:]

    pose = data.oMf[ee_frame_id]
    p = pose.translation
    R = pose.rotation

    p_err = p_des - p
    R_err = pin.log3(R_des@R.T)

    err = np.append(p_err, R_err)

    derr = dq_des - J@dq

    aq = np.linalg.pinv(J)@(ddq_des + kp*err + kd*derr - dJ@dq)

    u = np.array(data.M)@aq + data.nle
    
    return u


def main():
    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    
    print("\nRunning task space controller...")
    sim = Simulator(
        xml_path="./robots/universal_robots_ur5e/scene.xml",
        enable_task_space=True,
        show_viewer=True,
        record_video=True,
        video_path="./logs/videos/03_task_space.mp4",
        plots_path = "./logs/plots/",
        fps=30,
        width=1920,
        height=1080
    )
    sim.set_controller(task_space_controller)
    sim.run(time_limit=60.0)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model = pin.buildModelFromMJCF("./robots/universal_robots_ur5e/ur5e.xml")
    data = model.createData()


    main() 
