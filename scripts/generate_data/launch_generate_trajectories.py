import os
import socket

import numpy as np

from experiment_launcher import Launcher
from experiment_launcher.utils import is_local

########################################################################################################################
# EXPERIMENT PARAMETERS SETUP
# SELECT ONE

env_id, robot_id, num_contexts, num_trajectories_per_context, threshold_start_goal_pos, obstacle_cutoff_margin = 'EnvSimple2D', 'RobotPointMass', 500, 20, 1, 0.02
# env_id, robot_id, num_contexts, num_trajectories_per_context, threshold_start_goal_pos, obstacle_cutoff_margin = 'EnvNarrowPassageDense2D', 'RobotPointMass', 500, 20, 1, 0.02
# env_id, robot_id, num_contexts, num_trajectories_per_context, threshold_start_goal_pos, obstacle_cutoff_margin = 'EnvDense2D', 'RobotPointMass', 500, 20, 1, 0.02
# env_id, robot_id, num_contexts, num_trajectories_per_context, threshold_start_goal_pos, obstacle_cutoff_margin = 'EnvSpheres3D', 'RobotPanda', 500, 20, 1.83, 0.05  # 1.83 = 7 * np.deg2rad(15)


########################################################################################################################
# LAUNCHER

hostname = socket.gethostname()

LOCAL = is_local()
TEST = False
# USE_CUDA = True
USE_CUDA = False

N_SEEDS = num_contexts

N_EXPS_IN_PARALLEL = 15 if not USE_CUDA else 1

# N_CORES = N_EXPS_IN_PARALLEL
N_CORES = 8
MEMORY_SINGLE_JOB = 12000
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
PARTITION = 'gpu' if USE_CUDA else 'amd3,amd2,amd'
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1, gpu:rtx3090:1, gpu:a5000:1
CONDA_ENV = 'mpd-public'

exp_name = f'generate_trajectories'

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

launcher = Launcher(
    exp_name=exp_name,
    exp_file='generate_trajectories',
    # project_name='project01234',
    n_seeds=N_SEEDS,
    n_exps_in_parallel=N_EXPS_IN_PARALLEL,
    n_cores=N_CORES,
    memory_per_core=MEMORY_PER_CORE,
    days=0,
    hours=7,
    minutes=59,
    seconds=0,
    partition=PARTITION,
    conda_env=CONDA_ENV,
    gres=GRES,
    use_timestamp=True
)


########################################################################################################################
# RUN

launcher.add_experiment(
    env_id__=env_id,
    robot_id__=robot_id,

    num_trajectories=num_trajectories_per_context,

    threshold_start_goal_pos=threshold_start_goal_pos,
    obstacle_cutoff_margin=obstacle_cutoff_margin,

    device='cuda' if USE_CUDA else 'cpu',

    debug=False
)

launcher.run(LOCAL, TEST)
