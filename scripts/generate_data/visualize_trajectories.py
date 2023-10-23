import pickle

import matplotlib.pyplot as plt

import os.path

import torch
import yaml

from mpd.utils.loading import load_params_from_yaml
from torch_robotics import environments, robots
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

DATA_DIR = '../../data_trajectories/EnvSpheres3D-RobotPanda-cluster/66'

tensor_args = DEFAULT_TENSOR_ARGS

args = load_params_from_yaml(os.path.join(DATA_DIR, 'args.yaml'))

metadata = load_params_from_yaml(os.path.join(DATA_DIR, 'metadata.yaml'))
print(f"\n-------------- METADATA --------------")
print(yaml.dump(metadata))
print(f"\n--------------------------------------")
print()

# -------------------------------- Load env, robot, task ---------------------------------
# Environment
env_class = getattr(environments, args['env_id'])
env = env_class(tensor_args=tensor_args)

# Robot
robot_class = getattr(robots, args['robot_id'])
robot = robot_class(
    tensor_args=tensor_args
)

# Task
task = PlanningTask(
    env=env,
    robot=robot,
    obstacle_cutoff_margin=args['obstacle_cutoff_margin'],
    tensor_args=tensor_args
)

# -------------------------------- Load trajectories -------------------------
trajs_collision = torch.load(os.path.join(DATA_DIR, 'trajs-collision.pt')).to(**tensor_args)
trajs_free = torch.load(os.path.join(DATA_DIR, 'trajs-free.pt')).to(**tensor_args)

# trajs = torch.cat((trajs_collision, trajs_free))
trajs = trajs_free

# -------------------------------- Visualize ---------------------------------
planner_visualizer = PlanningVisualizer(task=task)

pos_trajs = robot.get_position(trajs)
start_state_pos = pos_trajs[0][0]
goal_state_pos = pos_trajs[0][-1]

planner_visualizer.plot_joint_space_state_trajectories(
    trajs=trajs,
    pos_start_state=start_state_pos, pos_goal_state=goal_state_pos,
    vel_start_state=torch.zeros_like(start_state_pos), vel_goal_state=torch.zeros_like(goal_state_pos),
)

plt.show()

planner_visualizer.render_robot_trajectories(
    trajs=trajs, start_state=start_state_pos, goal_state=goal_state_pos,
    render_planner=False,
)

plt.show()

planner_visualizer.animate_robot_trajectories(
    trajs=trajs, start_state=start_state_pos, goal_state=goal_state_pos,
    plot_trajs=True,
    video_filepath=os.path.join(DATA_DIR, 'robot-traj.mp4'),
    # n_frames=max((2, pos_trajs_iters[-1].shape[1]//10)),
    n_frames=trajs.shape[1],
    anim_time=args['duration']
)

