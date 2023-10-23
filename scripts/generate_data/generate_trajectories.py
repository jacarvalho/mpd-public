import os
import pickle
import time

import torch
import yaml
from matplotlib import pyplot as plt

from experiment_launcher import single_experiment_yaml, run_experiment
from experiment_launcher.utils import fix_random_seed
from mp_baselines.planners.gpmp2 import GPMP2
from mp_baselines.planners.hybrid_planner import HybridPlanner
from mp_baselines.planners.multi_sample_based_planner import MultiSampleBasedPlanner
from mp_baselines.planners.rrt_connect import RRTConnect
from torch_robotics import environments, robots
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer


def generate_collision_free_trajectories(
    env_id,
    robot_id,
    num_trajectories_per_context,
    results_dir,
    threshold_start_goal_pos=1.0,
    obstacle_cutoff_margin=0.03,
    n_tries=1000,
    rrt_max_time=300,
    gpmp_opt_iters=500,
    n_support_points=64,
    duration=5.0,
    tensor_args=None,
    debug=False,
):
    # -------------------------------- Load env, robot, task ---------------------------------
    # Environment
    env_class = getattr(environments, env_id)
    env = env_class(tensor_args=tensor_args)

    # Robot
    robot_class = getattr(robots, robot_id)
    robot = robot_class(tensor_args=tensor_args)

    # Task
    task = PlanningTask(
        env=env,
        robot=robot,
        obstacle_cutoff_margin=obstacle_cutoff_margin,
        tensor_args=tensor_args
    )

    # -------------------------------- Start, Goal states ---------------------------------
    start_state_pos, goal_state_pos = None, None
    for _ in range(n_tries):
        q_free = task.random_coll_free_q(n_samples=2)
        start_state_pos = q_free[0]
        goal_state_pos = q_free[1]

        if torch.linalg.norm(start_state_pos - goal_state_pos) > threshold_start_goal_pos:
            break

    if start_state_pos is None or goal_state_pos is None:
        raise ValueError(f"No collision free configuration was found\n"
                         f"start_state_pos: {start_state_pos}\n"
                         f"goal_state_pos:  {goal_state_pos}\n")

    n_trajectories = num_trajectories_per_context

    # -------------------------------- Hybrid Planner ---------------------------------
    # Sample-based planner
    rrt_connect_default_params_env = env.get_rrt_connect_params(robot=robot)
    rrt_connect_default_params_env['max_time'] = rrt_max_time

    rrt_connect_params = dict(
        **rrt_connect_default_params_env,
        task=task,
        start_state_pos=start_state_pos,
        goal_state_pos=goal_state_pos,
        tensor_args=tensor_args,
    )
    sample_based_planner_base = RRTConnect(**rrt_connect_params)
    # sample_based_planner_base = RRTStar(**rrt_connect_params)
    # sample_based_planner = sample_based_planner_base
    sample_based_planner = MultiSampleBasedPlanner(
        sample_based_planner_base,
        n_trajectories=n_trajectories,
        max_processes=-1,
        optimize_sequentially=True
    )

    # Optimization-based planner
    gpmp_default_params_env = env.get_gpmp2_params(robot=robot)
    gpmp_default_params_env['opt_iters'] = gpmp_opt_iters
    gpmp_default_params_env['n_support_points'] = n_support_points
    gpmp_default_params_env['dt'] = duration / n_support_points

    planner_params = dict(
        **gpmp_default_params_env,
        robot=robot,
        n_dof=robot.q_dim,
        num_particles_per_goal=n_trajectories,
        start_state=start_state_pos,
        multi_goal_states=goal_state_pos.unsqueeze(0),  # add batch dim for interface,
        collision_fields=task.get_collision_fields(),
        tensor_args=tensor_args,
    )
    opt_based_planner = GPMP2(**planner_params)

    ###############
    # Hybrid planner
    planner = HybridPlanner(
        sample_based_planner,
        opt_based_planner,
        tensor_args=tensor_args
    )

    # Optimize
    trajs_iters = planner.optimize(debug=debug, print_times=True, return_iterations=True)
    trajs_last_iter = trajs_iters[-1]

    # -------------------------------- Save trajectories ---------------------------------
    print(f'----------------STATISTICS----------------')
    print(f'percentage free trajs: {task.compute_fraction_free_trajs(trajs_last_iter)*100:.2f}')
    print(f'percentage collision intensity {task.compute_collision_intensity_trajs(trajs_last_iter)*100:.2f}')
    print(f'success {task.compute_success_free_trajs(trajs_last_iter)}')

    # save
    torch.cuda.empty_cache()
    trajs_last_iter_coll, trajs_last_iter_free = task.get_trajs_collision_and_free(trajs_last_iter)
    if trajs_last_iter_coll is None:
        trajs_last_iter_coll = torch.empty(0)
    torch.save(trajs_last_iter_coll, os.path.join(results_dir, f'trajs-collision.pt'))
    if trajs_last_iter_free is None:
        trajs_last_iter_free = torch.empty(0)
    torch.save(trajs_last_iter_free, os.path.join(results_dir, f'trajs-free.pt'))

    # save results data dict
    trajs_iters_coll, trajs_iters_free = task.get_trajs_collision_and_free(trajs_iters[-1])
    results_data_dict = {
        'duration': duration,
        'n_support_points': n_support_points,
        'dt': planner_params['dt'],
        'trajs_iters_coll': trajs_iters_coll.unsqueeze(0) if trajs_iters_coll is not None else None,
        'trajs_iters_free': trajs_iters_free.unsqueeze(0) if trajs_iters_free is not None else None,
    }

    with open(os.path.join(results_dir, f'results_data_dict.pickle'), 'wb') as handle:
        pickle.dump(results_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # -------------------------------- Visualize ---------------------------------
    planner_visualizer = PlanningVisualizer(task=task)

    trajs = trajs_last_iter_free
    pos_trajs = robot.get_position(trajs)
    start_state_pos = pos_trajs[0][0]
    goal_state_pos = pos_trajs[0][-1]

    fig, axs = planner_visualizer.plot_joint_space_state_trajectories(
        trajs=trajs,
        pos_start_state=start_state_pos, pos_goal_state=goal_state_pos,
        vel_start_state=torch.zeros_like(start_state_pos), vel_goal_state=torch.zeros_like(goal_state_pos),
    )

    # save figure
    fig.savefig(os.path.join(results_dir, f'trajectories.png'), dpi=300)
    plt.close(fig)

    num_trajectories_coll, num_trajectories_free = len(trajs_last_iter_coll), len(trajs_last_iter_free)
    return num_trajectories_coll, num_trajectories_free


@single_experiment_yaml
def experiment(
    # env_id: str = 'EnvDense2D',
    # env_id: str = 'EnvSimple2D',
    # env_id: str = 'EnvNarrowPassageDense2D',
    env_id: str = 'EnvSpheres3D',

    # robot_id: str = 'RobotPointMass',
    robot_id: str = 'RobotPanda',

    n_support_points: int = 64,
    duration: float = 5.0,  # seconds

    # threshold_start_goal_pos: float = 1.0,
    threshold_start_goal_pos: float = 1.83,

    obstacle_cutoff_margin: float = 0.05,

    num_trajectories: int = 5,

    # device: str = 'cpu',
    device: str = 'cuda',

    debug: bool = True,

    #######################################
    # MANDATORY
    seed: int = int(time.time()),
    # seed: int = 0,
    # seed: int = 1679258088,
    results_dir: str = f"data",

    #######################################
    **kwargs
):
    if debug:
        fix_random_seed(seed)

    print(f'\n\n-------------------- Generating data --------------------')
    print(f'Seed:  {seed}')
    print(f'Env:   {env_id}')
    print(f'Robot: {robot_id}')
    print(f'num_trajectories: {num_trajectories}')

    ####################################################################################################################
    tensor_args = {'device': device, 'dtype': torch.float32}

    metadata = {
        'env_id': env_id,
        'robot_id': robot_id,
        'num_trajectories': num_trajectories
    }
    with open(os.path.join(results_dir, 'metadata.yaml'), 'w') as f:
        yaml.dump(metadata, f, Dumper=yaml.Dumper)

    # Generate trajectories
    num_trajectories_coll, num_trajectories_free = generate_collision_free_trajectories(
        env_id,
        robot_id,
        num_trajectories,
        results_dir,
        threshold_start_goal_pos=threshold_start_goal_pos,
        obstacle_cutoff_margin=obstacle_cutoff_margin,
        n_support_points=n_support_points,
        duration=duration,
        tensor_args=tensor_args,
        debug=debug,
    )

    metadata.update(
        num_trajectories_generated=num_trajectories_coll + num_trajectories_free,
        num_trajectories_generated_coll=num_trajectories_coll,
        num_trajectories_generated_free=num_trajectories_free,
    )
    with open(os.path.join(results_dir, 'metadata.yaml'), 'w') as f:
        yaml.dump(metadata, f, Dumper=yaml.Dumper)


if __name__ == '__main__':
    run_experiment(experiment)
