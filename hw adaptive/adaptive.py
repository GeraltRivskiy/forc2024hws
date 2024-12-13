

import numpy as np
from simulator import Simulator
from pathlib import Path
from typing import Dict
import os 
import pinocchio as pin
import matplotlib.pyplot as plt

np.random.seed(42)


def joint_inv_din_controller(self, q: np.ndarray, dq: np.ndarray, ddq, t: float) -> np.ndarray:
    """Joint space PD controller.
    
    Args:
        q: Current joint positions [rad]
        dq: Current joint velocities [rad/s]
        t: Current simulation time [s]
        
    Returns:
        tau: Joint torques command [Nm]
    """

    global q_fact, times, q_des
    qd, dqd = trajectory(t)
    if self.name == 'sim':
        q_fact['inv_din'] = np.vstack([q_fact['inv_din'], q])
        q_des['inv_din'] = np.vstack([q_des['inv_din'], qd])
        errors['inv_din'] = np.vstack([errors['inv_din'], q-qd])
        times['inv_din'] = np.append(times['inv_din'], t)
    elif self.name == 'sim_add_mass':
        q_fact['inv_din_added_mass'] = np.vstack([q_fact['inv_din_added_mass'], q])
        q_des['inv_din_added_mass'] = np.vstack([q_des['inv_din_added_mass'], qd])
        errors['inv_din_added_mass'] = np.vstack([errors['inv_din_added_mass'], q-qd])
        times['inv_din_added_mass'] = np.append(times['inv_din_added_mass'], t)

    pin.computeAllTerms(model, data, q, dq)
    M, nle = data.M, data.nle
    kp = np.array([50, 50, 25, 80, 80, 120])
    kd = np.array([35, 35, 10, 20, 20, 4])
    tau = M@(kp * (qd - q) - kd * (dq - dqd)) + nle
    return tau

def joint_adaptive_controller(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray, t: float) -> np.ndarray:
    """Example task space controller."""
    global th, times, q_fact, q_des
    # kp = np.array([10, 10, 10, 10, 10, 10])
    # kd = np.array([3, 3, 3, 2, 2, 2])

    kd = np.array([100, 100, 100, 4, 0.2, 0.2])
    ddq_des = np.array([0, 0, 0, 0, 0, 0])

    qd, dqd = trajectory(t)
    # qd = np.array([-1.4, -1.3, 1, 0, 0, 0])
    q_fact['adaptive_added_mass'] = np.vstack([q_fact['adaptive_added_mass'], q])
    q_des['adaptive_added_mass'] = np.vstack([q_des['adaptive_added_mass'], qd])
    times['adaptive_added_mass'] = np.append(times['adaptive_added_mass'], t)
    errors['adaptive_added_mass'] = np.vstack([errors['adaptive_added_mass'], q-qd])

    Lambda = np.diag([1, 1, 1, 1.1, 1, 20])
    Gamma = np.diag([2500]*60)

    pin.computeAllTerms(model, data, q, dq)
    r = (dq - dqd) + Lambda@(q-qd) 

    Y = pin.computeJointTorqueRegressor(model, data, q, dq, ddq_des)

    u = Y@th - kd*r
    dth = -np.linalg.inv(Gamma)@Y.T@r
    th +=dth
    
    return u


def main():
    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    
    print("\nRunning task space controller...")
    sim = Simulator(
        xml_path="./robots/universal_robots_ur5e/scene.xml",
        enable_task_space=False,
        show_viewer=True,
        record_video=True,
        video_path="./logs/videos/point_inv_din.mp4",
        plots_path = "./logs/plots/",
        fps=30,
        width=1920,
        height=1080,
        name = "sim"
    )
    sim.set_controller(joint_inv_din_controller)
    sim.run(time_limit=15.0)

    sim_add_mass = Simulator(
        xml_path="./robots/universal_robots_ur5e/scene.xml",
        enable_task_space=False,
        show_viewer=True,
        record_video=True,
        video_path="./logs/videos/point_inv_din_add_mass.mp4",
        plots_path = "./logs/plots/",
        fps=30,
        width=1920,
        height=1080,
        name = "sim_add_mass",
        added_mass=added_mass
    )
    sim_add_mass.set_controller(joint_inv_din_controller)
    sim_add_mass.run(time_limit=15.0)

    sim_adaptive = Simulator(
        xml_path="./robots/universal_robots_ur5e/scene.xml",
        enable_task_space=False,
        show_viewer=True,
        record_video=True,
        video_path="./logs/videos/point_adaptive_add_mass_noise.mp4",
        plots_path = "./logs/plots/",
        fps=30,
        width=1920,
        height=1080,
        name = "sim_adaptive",
        added_mass=added_mass
    )
    sim_adaptive.set_controller(joint_adaptive_controller)
    sim_adaptive.run(time_limit=15.0)



def plot_results(times, q_fact, qd, errors, name):
    global added_mass
    colors = ['green', 'blue', 'red', 'orange', 'gray', 'purple']

    plt.figure(figsize=(15, 10))
    for i in range(qd.shape[1]):
        plt.plot(times, qd[:,i], label=f'Joint {i+1} desired', linestyle='dashed', color=colors[i])
        plt.plot(times, q_fact[:, i], label=f'Joint {i+1}', color=colors[i])
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Positions [rad]')
    plt.legend()
    plt.grid(True)
    if name == "inv_din":
        plt.title(f'[{t_name}] Joint Positions over Time') 
    else:
        plt.title(f'[{t_name}] Joint Positions over Time, added mass = {added_mass}')
    plt.savefig(f'./logs/plots/point_{name}.png')

    plt.figure(figsize=(15, 10))
    for i in range(qd.shape[1]):
        plt.plot(times, errors[:, i], label=f'Joint {i+1}', color=colors[i])
    plt.xlabel('Time [s]')
    plt.ylabel(f'[{t_name}] Positions error [rad]')
    plt.legend()
    plt.grid(True)
    if name == "inv_din":
        plt.title(f'[{t_name}] Positions error over Time') 
    else:
        plt.title(f'[{t_name}] Positions error over Time, added mass = {added_mass}')
    plt.savefig(f'./logs/plots/point_{name}_errors.png')

    plt.close()

def add_mass(model: pin.Model, added_mass, eeid):
    new_inertia = pin.Inertia(mass=added_mass,
                              lever=model.inertias[eeid].lever,  # Оставляем прежний центр масс
                              inertia=model.inertias[eeid].inertia  # Оставляем прежний тензор инерции
                              )

    model.inertias[eeid] = new_inertia
    return model

def trajectory(t):
    # qd = np.array([-1.5 + 0.25*np.sin(t),
    #                -1.2 + 0.25*np.sin(t),
    #                 0.7 + 0.25*np.sin(t),
    #                 0.5 + 0.25*np.sin(t),
    #                -0.5 + 0.25*np.sin(t),
    #                 0.25*np.sin(2*t)])
    # dqd = np.array([0.25*np.cos(t),
    #                 0.25*np.cos(t),
    #                 0.25*np.cos(t),
    #                 0.25*np.cos(t),
    #                 0.25*np.cos(t),
    #                 0.25*np.cos(t)])
    qd = np.array([-1.4,
                   -1.3,
                    1,
                    0,
                    0,
                    0])
    dqd = np.array([0,
                    0,
                    0,
                    0,
                    0,
                    0])
    return qd, dqd

if __name__ == "__main__":
    added_mass = 1
    model = pin.buildModelFromMJCF("./robots/universal_robots_ur5e/ur5e.xml")
    data = model.createData()
    plt.rcParams.update({'font.size': 17})
    th = np.empty((0,))
    for i in range(len(model.inertias[1:])):
        th = np.append(th, model.inertias[i+1].toDynamicParameters())
    noise = np.random.uniform(-0.2, 0.2, th.shape) * th
    th += noise

    sim_types = ['inv_din', 'inv_din_added_mass', 'adaptive_added_mass']
    times, q_fact, q_des, errors = {}, {}, {}, {}
    for t_name in sim_types:
        times [t_name]= np.empty((0,))
        q_fact [t_name] = np.empty((0,6))
        q_des [t_name]= np.empty((0,6))
        errors [t_name] = np.empty((0,6))

    main() 

    for t_name in sim_types:
        plot_results(times[t_name], q_fact[t_name], q_des[t_name], errors[t_name], t_name)

