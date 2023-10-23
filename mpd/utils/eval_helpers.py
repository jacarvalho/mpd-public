import time

import numpy as np
import torch
from einops import repeat, rearrange
from sklearn.cluster import KMeans
import math

import os
import pandas
def plot_env_image(ax=None, env_image=None):
    ax.imshow(env_image.permute(1, 2, 0), origin="lower")


def plot_trajs(trajs, collisions=None, task_context=None, ax=None, scale=None, color=None, best_index=None,
               linewidth=6, linestyle='-', **kwargs):
    if task_context is not None:
        if scale is not None:
            scaled_task_context = ((task_context + 1) / 2) * scale
        else:
            scaled_task_context = task_context
        ax.scatter(scaled_task_context[0], scaled_task_context[1], color='red', zorder=20, marker='o')
        ax.scatter(scaled_task_context[2], scaled_task_context[3], color='red', zorder=20, marker='x')

    if scale is not None:
        trajs = ((trajs + 1) / 2) * scale  # Scale to image size

    for i, traj in enumerate(trajs):
        if collisions is not None:
            collides = collisions[i]
        else:
            collides = False

        if best_index is not None:
            is_best = i == best_index
            kwargs['alpha'] =1 if is_best else 0.5,
        else:
            is_best = False

        if is_best:
            kwargs.update(zorder=10)
        ax.plot(traj[:, 0], traj[:, 1],
                linestyle='--' if collides else linestyle, color=color,
                linewidth=6 if is_best else linewidth,
                **kwargs)
        #ax.scatter(traj[:, 0], traj[:, 1])

def plot_trajs_3d(trajs, collisions=None, task_context=None, ax=None, scale=None, color=None, best_index=None,
               linewidth=6, linestyle='-', **kwargs):

    if task_context is not None:
        if scale is not None:
            scaled_task_context = ((task_context + 1) / 2) * scale
        else:
            scaled_task_context = task_context
        ax.scatter(scaled_task_context[0], scaled_task_context[1], scaled_task_context[2], color='red', zorder=20, marker='o')
        ax.scatter(scaled_task_context[3], scaled_task_context[4], scaled_task_context[5],color='red', zorder=20, marker='x')

    if scale is not None:
        trajs = ((trajs + 1) / 2) * scale  # Scale to image size

    for i, traj in enumerate(trajs):
        if collisions is not None:
            collides = collisions[i]
        else:
            collides = False

        if best_index is not None:
            is_best = i == best_index
        else:
            is_best = False

        if is_best:
            kwargs.update(zorder=10)
        ax.plot(traj[:, 0], traj[:, 1],traj[:, 2],
                linestyle='--' if collides else linestyle, color=color,
                linewidth=6 if is_best else linewidth,
                alpha=1 if is_best else 0.5,
                **kwargs)


def sample_trajs(sampler,
                 model,
                 env_features=None,
                 task_context=None,
                 task_field=None,
                 device=None,
                 num_samples=5,
                 **kwargs
                 ):
    # Trajectory
    batch_size = num_samples
    env_features_d_batch = {}
    if env_features:
        env_features_batch = repeat(env_features, '1 d -> b d', b=batch_size)
        env_features_d_batch = {model.env_model.output_field: env_features_batch}
    task_c_batch = repeat(task_context, 'd -> b d', b=batch_size).to(device)
    samples = sampler.sample(
        model,
        context={**env_features_d_batch,
                 task_field: task_c_batch,
                 },
        batch_size=batch_size,
        **kwargs
    )[model.input_field]

    return samples


def get_best_index(torch_trajs=None, collisions=None):
    idx_no_collision = torch.where(1 - torch.Tensor(collisions))[0]
    if len(idx_no_collision) != 0:
        feasible_samples = torch_trajs[idx_no_collision]
        n_support_pointsgths = torch.linalg.norm(torch.diff(feasible_samples, dim=(-2)), dim=(-1)).sum(dim=-1)

        idx_best_traj = torch.argmin(n_support_pointsgths).item()
        return idx_no_collision[idx_best_traj].item()
    else:
        idx_random = np.random.choice(np.arange(torch_trajs.shape[0]))
        return idx_random


def get_best_index_by_sdf(torch_trajs=None, env_features=None, model=None, device=None):
    # SDF
    sdf_locations = rearrange(torch_trajs, 'b h d -> (b h) d')
    env_features_sdf = repeat(env_features, '1 d -> b d', b=sdf_locations.shape[0])
    model_output = model.compute_sdf({
        model.sdf_model.input_field: env_features_sdf.to(device),
        model.sdf_model.sdf_location_field: sdf_locations.to(device)
    })
    sdfs_flat = model_output['sdf']

    costs = torch.where(sdfs_flat < 0, sdfs_flat, torch.zeros_like(sdfs_flat))
    costs = rearrange(costs, '(b h) d -> b h d', b=torch_trajs.shape[0], h=torch_trajs.shape[1])

    idx_feasible_samples = torch.argwhere(costs.sum(dim=(-2, -1)) == 0)
    if idx_feasible_samples.nelement() != 0:
        feasible_samples = torch_trajs[idx_feasible_samples]
        n_support_pointsgths = torch.linalg.norm(torch.diff(feasible_samples, dim=(-2)), dim=(-1)).sum(dim=-1)

        idx_best_traj = torch.argmin(n_support_pointsgths).item()
        return idx_feasible_samples[idx_best_traj].item()
    else:
        idx_random = np.random.choice(np.arange(torch_trajs.shape[0]))
        return idx_random


def k_means_select_k(X: np.ndarray, k_range: np.ndarray) -> int:
    # selects the number of clusters in Kmeans using the Elbow method
    # https://www.codecademy.com/learn/machine-learning/modules/dspath-clustering/cheatsheet
    # https://stackoverflow.com/questions/72236122/elbow-method-for-k-means-in-python
    wss = np.empty(k_range.size)
    for i, k in enumerate(k_range):
        kmeans = KMeans(n_clusters=k, init='k-means++')
        kmeans.fit(X)
        wss[i] = kmeans.inertia_
        # wss[i] = ((X - kmeans.cluster_centers_[kmeans.labels_]) ** 2).sum()

    # computes the elbow of a curve
    slope = (wss[0] - wss[-1]) / (k_range[0] - k_range[-1])
    intercept = wss[0] - slope * k_range[0]
    y = k_range * slope + intercept

    return k_range[(y - wss).argmax()]


def evaluation_metrics_rrt_variable_horizons(
        trajs_list,
        print_info=True,
        print_label='RRT_connect',
):
    metrics = {}
    # by design rrt doesn't have collisions
    metrics['percentage_coll_free_trajs'] = 100.
    metrics['percentage_in_collision'] = 0.

    traj_distances = []
    acceleration = []
    cos_sims = []
    for traj in trajs_list:
        finite_diff = np.diff(traj, axis=-2)
        traj_distance = np.sum(np.linalg.norm(finite_diff, axis=-1), axis=-1)

        # Remove duplicates after calculating distance.
        # Otherwise mean is thrown off. This is reasonable because we could interpolate infinitely many points the get
        # a perfect score
        if len(finite_diff) > 1:
            finite_diff = purge_duplicates_from_traj(finite_diff)

        # remove straight points
        if finite_diff.shape[0] < 2:
            accelerations = 0
            cos_sims.append(0)
        else:
            accelerations = np.linalg.norm(np.diff(finite_diff, axis=-2), axis=-1)
            accelerations = accelerations.mean()

            cos_sim = np.zeros((finite_diff.shape[0] - 1))
            for i in range(finite_diff.shape[0] - 1):
                x1 = finite_diff[i]
                x2 = finite_diff[i + 1]

                sim = 1 - (np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2)))
                cos_sim[i] = sim

            cos_sims.append(cos_sim.mean())
        acceleration.append(accelerations)
        #if len(accelerations) > 0:
        #    if len(accelerations) == 1:
        #        smoothness_costs.append(accelerations.item())
        #    else:
        #        smoothness_costs += list(accelerations)
        traj_distances.append(traj_distance)

        #smoothness_cost = traj_distance / len(traj)
        #smoothness_costs.append(smoothness_cost)
    average_acceleration = np.mean(acceleration)
    average_traj_distance = np.mean(traj_distances)
    average_cosine_sim = np.mean(cos_sims)
    metrics['average_acceleration'] = average_acceleration
    metrics['average_distance'] = average_traj_distance
    metrics['average_cosine_sim'] = average_cosine_sim

    return metrics

def evaluation_metrics(
        trajs_torch,
        trajs_coll_free_torch,
        print_info=True,
        print_label='StochGPMP',
        simple_metrics=False
):
    trajs_np = to_numpy(trajs_torch)
    trajs_coll_free_np = to_numpy(trajs_coll_free_torch)

    metrics = {}

    B, H, D = trajs_torch.shape
    B_coll_free, _, _ = trajs_coll_free_torch.shape

    # 3. Number of collision free trajectories
    percentage_coll_free_trajs = B_coll_free / B * 100
    metrics['percentage_coll_free_trajs'] = percentage_coll_free_trajs

    # 4. Average smoothness
    distance = torch.linalg.norm(torch.diff(trajs_torch, dim=-2), dim=-1)
    smoothness_cost = distance.sum(dim=-1)
    smoothness_cost *= 1 / H
    average_acceleration = smoothness_cost.mean()
    average_distance = distance.sum(dim=-1).mean()

    finite_diff = torch.diff(trajs_torch, dim=-2)
    accelerations = torch.linalg.norm(torch.diff(torch.diff(trajs_torch, dim=-2), dim=-2), dim=-1)
    #smoothness_costs.append(accelerations)

    cos_sim = np.zeros((finite_diff.shape[0], finite_diff.shape[1]-1))
    for b, batch in enumerate(finite_diff):
        for i in range(batch.shape[0]-1):
            x1 = finite_diff[b, i].unsqueeze(0)
            x2 = finite_diff[b, i+1].unsqueeze(0)
            sim = 1 - to_numpy(torch.nn.functional.cosine_similarity(x1, x2))
            cos_sim[b][i] = sim

    metrics['average_acceleration'] = to_numpy(accelerations)
    metrics['average_cosine_sim'] = cos_sim.mean()

    metrics['average_distance'] = to_numpy(average_distance)

    if not simple_metrics:
        # 1. Mode discovery - clustering (flatten the trajectories)
        k_opt_trajs = k_means_select_k(trajs_np.reshape(B, H * D), np.arange(1, np.min((21, B))))
        metrics['k_opt_trajs'] = k_opt_trajs
        metrics['k_opt_trajs_coll_free'] = 0
        if B_coll_free == 1:
            metrics['k_opt_trajs_coll_free'] = 1
        if B_coll_free > 1:
            k_opt_trajs_coll_free = k_means_select_k(trajs_coll_free_np.reshape(B_coll_free, H * D),
                                                     np.arange(1, np.min((21, B_coll_free))))
            metrics['k_opt_trajs_coll_free'] = k_opt_trajs_coll_free

        # 2. Spatial coverage - variance for each time step
        # https://online.stat.psu.edu/stat505/lesson/1/1.5
        for trajs, trajs_label in zip([trajs_torch, trajs_coll_free_torch], ['trajs', 'trajs_coll_free']):
            spatial_coverage_trace_average = 0
            spatial_coverage_determinant_average = 0

            if trajs.shape[0] > 0:
                covar_trajs = batch_cov(rearrange(trajs, ' b h d -> h b d'))
                spatial_coverage_trace_all_steps = batch_trace(covar_trajs)
                spatial_coverage_trace_average = torch.mean(spatial_coverage_trace_all_steps)
                spatial_coverage_determinant_all_steps = torch.det(covar_trajs)
                spatial_coverage_determinant_average = torch.mean(spatial_coverage_determinant_all_steps)

            metrics[f'spatial_coverage_trace_average_{trajs_label}'] = spatial_coverage_trace_average
            metrics[f'spatial_coverage_determinant_average_{trajs_label}'] = spatial_coverage_determinant_average

    if print_info:
        print(f'---- Trajectories ALL----')
        print(f'{print_label} number of mean_trajs: {B}')
        print(f'{print_label} number of coll_free_trajs: {B_coll_free}')
        print(f'{print_label} percentage coll_free: {percentage_coll_free_trajs:.2f}')
        print(f'{print_label} average smoothness cost: {average_acceleration:.2f}')
        if not simple_metrics:

            print()
            print(f'{print_label} k_opt_trajs: {metrics["k_opt_trajs"]}')
            print(
                f'{print_label} spatial_coverage_trace_average_trajs: {metrics["spatial_coverage_trace_average_trajs"]:.5f}')
            print(
                f'{print_label} spatial_coverage_determinant_average_trajs: {metrics["spatial_coverage_determinant_average_trajs"]:.5f}')
            print(f'----Trajectories Collision Free----')
            print(f'{print_label} k_opt_trajs_coll_free: {metrics["k_opt_trajs_coll_free"]}')
            print(
                f'{print_label} spatial_coverage_trace_average_trajs_coll_free: {metrics["spatial_coverage_trace_average_trajs_coll_free"]:.5f}')
            print(
                f'{print_label} spatial_coverage_determinant_average_trajs: {metrics["spatial_coverage_determinant_average_trajs_coll_free"]:.5f}')

    return metrics


def eval_sbm(model, sampler=None, extra_objective_fn_grad=None, extra_objective_fn_grad_cascading=None,
             task_context=None, task_field=None, num_samples=None, device=None, obst_map=None, debug=False,
             print_label='SBM', env=None, use_env_collision=False,
             ):
    # Sample trajectories
    s = time.time()
    sbm_trajs_torch = sample_trajs(
        sampler,
        model,
        task_context=task_context,
        task_field=task_field,
        num_samples=num_samples,
        device=device,
        extra_objective_fn_grad_cascading=extra_objective_fn_grad_cascading,
        extra_objective_fn_grad=extra_objective_fn_grad,
    )
    sbm_sampling_time = time.time() - s
    if use_env_collision:

        sbm_coll_free_trajs_torch, percent_in_coll = compute_coll_free_trajs_env(sbm_trajs_torch, env, return_coll_stat=True)
    else:
        sbm_coll_free_trajs_torch, percent_in_coll = compute_coll_free_trajs_obst_map(sbm_trajs_torch, obst_map,
                                                                                      return_coll_stat=True)


    sbm_metrics = dict()
    sbm_metrics['sampling_time'] = sbm_sampling_time
    if debug:
        print(f'{print_label} sampling time: {sbm_sampling_time:.3f}')
    sbm_metrics['percentage_in_collision'] = percent_in_coll.item()
    eval_metrics = evaluation_metrics(sbm_trajs_torch, sbm_coll_free_trajs_torch, print_info=debug,
                                      print_label=print_label,
                                      simple_metrics=True)
    sbm_metrics.update(**eval_metrics)
    return sbm_metrics, sbm_trajs_torch, sbm_coll_free_trajs_torch

def eval_2D_stoch_gpmp(partial_stochgpmp_params, start_state, goal_state, tensor_args=None, initial_particle_means=None,
                    obst_map=None, print_label='StochGPMP', debug=False, env=None, use_env_collision=False):
    stochgpmp_params = {
        **partial_stochgpmp_params,
        "start_state": torch.cat((start_state, torch.zeros(start_state.shape[0], **tensor_args))),
        "multi_goal_states": torch.cat((goal_state, torch.zeros(goal_state.shape[0], **tensor_args))).unsqueeze(0)
    }
    # Works so don't use new for 2D
    stochgpmp_coll_free_trajs_torch, stochgpmp_mean_trajs_torch, _, stochgpmp_metrics = generate_stochgpmp_trajs(
        stochgpmp_params, obst_map, initial_particle_means=initial_particle_means, get_statistics=True, env=env,
        use_env_collision=use_env_collision, use_legacy=True)

    eval_metrics = evaluation_metrics(stochgpmp_mean_trajs_torch,
                                      stochgpmp_coll_free_trajs_torch, print_info=debug,
                                      print_label=print_label, simple_metrics=True)
    stochgpmp_metrics.update(**eval_metrics)
    # TODO debug print
    return stochgpmp_metrics, stochgpmp_mean_trajs_torch, stochgpmp_coll_free_trajs_torch
def eval_3D_stoch_gpmp(partial_stochgpmp_params, start_state, goal_state, tensor_args=None, initial_particle_means=None,
                       obst_map=None, print_label='StochGPMP', debug=False, env=None, use_env_collision=False):
    start_state = torch.cat((start_state, torch.zeros(start_state.shape[0], **tensor_args)))
    multi_goal_state = torch.cat((goal_state, torch.zeros(goal_state.shape[0], **tensor_args))).unsqueeze(0)
    stochgpmp_params = {
        **partial_stochgpmp_params,
        "start_state": start_state,
        "multi_goal_states": multi_goal_state,
        "tensor_args": tensor_args,
    }

    n_dof = stochgpmp_params['n_dof']
    n_support_points = stochgpmp_params['n_support_points']
    dt = stochgpmp_params['dt']
    multi_goal_states = stochgpmp_params['multi_goal_states']
    num_particles_per_goal = stochgpmp_params['num_particles_per_goal']
    num_samples = stochgpmp_params['num_samples']
    sigma_start = stochgpmp_params.pop('sigma_start')
    sigma_gp = stochgpmp_params.pop('sigma_gp')
    sigma_goal = stochgpmp_params.pop('sigma_goal')
    sigma_obst = stochgpmp_params.pop('sigma_obst')

    cost_sigmas = dict(
        sigma_start=sigma_start,
        sigma_gp=sigma_gp,
    )
    sigma_coll = sigma_obst
    sigma_goal_prior = sigma_goal

    # Construct cost function
    cost_prior = CostGP(
        n_dof, n_support_points, start_state, dt,
        cost_sigmas, tensor_args
    )
    cost_goal_prior = CostGoalPrior(n_dof, n_support_points, multi_goal_states=multi_goal_states, 
                                    num_particles_per_goal=num_particles_per_goal, 
                                    num_samples=num_samples, 
                                    sigma_goal_prior=sigma_goal_prior,
                                    tensor_args=tensor_args)
    cost_obst_2D = CostCollision(n_dof, n_support_points, field=obst_map, sigma_coll=sigma_coll)
    cost_func_list = [cost_prior, cost_goal_prior, cost_obst_2D]
    cost_composite = CostComposite(n_dof, n_support_points, cost_func_list)

    stochgpmp_params['cost'] = cost_composite

    stochgpmp_coll_free_trajs_torch, stochgpmp_mean_trajs_torch, _, stochgpmp_metrics = generate_stochgpmp_trajs(
        stochgpmp_params, obst_map, initial_particle_means=initial_particle_means, get_statistics=True, env=env,
        use_env_collision=use_env_collision)

    eval_metrics = evaluation_metrics(stochgpmp_mean_trajs_torch,
                                      stochgpmp_coll_free_trajs_torch, print_info=debug,
                                      print_label=print_label, simple_metrics=True)
    stochgpmp_metrics.update(**eval_metrics)
    # TODO debug print
    return stochgpmp_metrics, stochgpmp_mean_trajs_torch, stochgpmp_coll_free_trajs_torch

def eval_panda_stoch_gpmp(partial_stochgpmp_params, start_state, goal_state, tensor_args=None, initial_particle_means=None,
                    print_label='StochGPMP', debug=False, env=None):
    start_state = torch.cat((start_state, torch.zeros(start_state.shape[0], **tensor_args)))
    multi_goal_state = torch.cat((goal_state, torch.zeros(goal_state.shape[0], **tensor_args))).unsqueeze(0)
    stochgpmp_params = {
        **partial_stochgpmp_params,
        "start_state": start_state,
        "multi_goal_states": multi_goal_state
    }
    n_dof = stochgpmp_params['n_dof']
    n_support_points = stochgpmp_params['n_support_points']
    dt = stochgpmp_params['dt']
    device=tensor_args['device']
    statistics = dict()
    frame = env.pos_frame(env.joint_to_task(multi_goal_state[0])[:, :, -1])
    target_H = frame.get_transform_matrix()
    #_, target_H = env.task_to_joint(env.joint_to_task(multi_goal_state[0]))
    panda_floor = FloorDistanceField(margin=0.05, device=device)
    panda_self_link = LinkSelfDistanceField(margin=0.03, device=device)
    panda_collision_link = LinkDistanceField(device=device)
    panda_goal = EESE3DistanceField(target_H, device=device)

    # Factored Cost params
    prior_sigmas = dict(
        sigma_start=0.0001,
        sigma_gp=0.0007,
    )
    sigma_floor = 0.1
    sigma_self = 0.01
    sigma_coll = 0.0007
    sigma_goal = 0.00007
    sigma_goal_prior = 20.

    # Construct cost function
    cost_prior = CostGP(
        n_dof, n_support_points, start_state, dt,
        prior_sigmas, tensor_args
    )
    #cost_floor = CostCollision(n_dof, n_support_points, field=panda_floor, sigma_coll=sigma_floor)
    cost_self = CostCollision(n_dof, n_support_points, field=panda_self_link, sigma_coll=sigma_self)
    cost_coll = CostCollision(n_dof, n_support_points, field=panda_collision_link, sigma_coll=sigma_coll)
    cost_goal = CostGoal(n_dof, n_support_points, field=panda_goal, sigma_goal=sigma_goal)
    cost_goal_prior = CostGoalPrior(n_dof, n_support_points, multi_goal_states=stochgpmp_params['multi_goal_states'],
                                    num_particles_per_goal=stochgpmp_params['num_particles_per_goal'],
                                    num_samples=stochgpmp_params['num_samples'],
                                    sigma_goal_prior=sigma_goal_prior,
                                    tensor_args=tensor_args)
    cost_func_list = [cost_prior, cost_goal_prior, cost_self, cost_coll, cost_goal]
    # cost_func_list = [cost_prior, cost_goal_prior, cost_goal]
    cost_composite = CostComposite(n_dof, n_support_points, cost_func_list, FK=env.diff_panda.compute_forward_kinematics_all_links)

    stochgpmp_params['cost'] = cost_composite
    obs = {
        'obstacle_spheres': env.sphere_approximation
    }
    stochgpmp_coll_free_trajs_torch, stochgpmp_mean_trajs_torch, _, stochgpmp_metrics = generate_stochgpmp_trajs(
        stochgpmp_params, initial_particle_means=initial_particle_means, get_statistics=True, env=env,
        use_env_collision=True, obs=obs)

    eval_metrics = evaluation_metrics(stochgpmp_mean_trajs_torch,
                                      stochgpmp_coll_free_trajs_torch, print_info=debug,
                                      print_label=print_label, simple_metrics=True)
    stochgpmp_metrics.update(**eval_metrics)

    return stochgpmp_metrics, stochgpmp_mean_trajs_torch, stochgpmp_coll_free_trajs_torch

    planner = StochGPMP(**stochgpmp_params, initial_particle_means=initial_particle_means)

    s = time.time()
    for i in range(partial_stochgpmp_params['opt_iters'] + 1):
        print(i)
        time_start = time.time()
        planner.optimize(obs)
        print(f'Time(s) per iter: {time.time() - time_start} sec')
        controls, _, trajectories, trajectory_means, weights = planner.get_recent_samples()
    statistics['sampling_time'] = time.time() - s

    stochgpmp_coll_free_trajs_torch, stochgpmp_mean_trajs_torch, _, stochgpmp_metrics = generate_stochgpmp_trajs(
        stochgpmp_params, initial_particle_means=initial_particle_means, get_statistics=True, env=env,
        use_env_collision=True)


    free_trajs, percent_in_coll = compute_coll_free_trajs_env(trajectory_means, env, return_coll_stat=True)
    statistics['percentage_in_collision'] = to_numpy(percent_in_coll)
    eval_metrics = evaluation_metrics(trajectory_means,
                                      free_trajs, print_info=debug,
                                      print_label=print_label, simple_metrics=True)
    statistics.update(**eval_metrics)
    # TODO debug print
    return statistics, trajectory_means, free_trajs


def eval_rrt(env, start_state, goal_state, num_samples=None, variant='connect',max_step=0.07, max_iterations=1000, debug=False):
    start = to_numpy(start_state)
    goal = to_numpy(goal_state)
    trajs = []
    # Sample trajectories
    s = time.time()
    for i in range(num_samples):
        for t in range(100): # try 100 times
            print(t)
            if variant == 'connect':
                path = rrt_connect(start, goal, distance_fn=env.distance_fn,
                                   sample_fn=env.sample_fn,
                                   extend_fn=env.wrap_extend_fn(max_step=max_step, max_dist=100),
                                   collision_fn=env.collision_fn, max_iterations=max_iterations)
            elif variant == 'star':
                path = rrt_star(start, goal, distance_fn=env.distance_fn,
                                   sample_fn=env.sample_fn,
                                   extend_fn=env.extend_fn,
                                   collision_fn=env.collision_fn, max_iterations=max_iterations, radius=0.1, debug=debug)
            if path is not None:
                break

        if path is None:
            exit(f'RRT {variant} couldn\'t find a path')
        trajs.append(np.array(path))
    # TODO makes times per traj
    rrt_sampling_time = time.time() - s
    rrt_metrics = dict()
    rrt_metrics['sampling_time'] = rrt_sampling_time/num_samples
    rrt_metrics['percentage_in_collision'] = 0
    #rrt_trajs_np = np.array(trajs)
    eval_metrics = evaluation_metrics_rrt_variable_horizons(trajs, print_info=True, print_label=f'RRT_{variant}')
    rrt_metrics.update(**eval_metrics)
    return rrt_metrics, trajs, trajs  # keep same interface

def save_metrics(results_all_contexts, results_dir, round_to=2):
    # Average results for all contexts
    pd_results_all_contexts = pandas.DataFrame.from_dict(results_all_contexts).transpose()
    pd_results_all_contexts_mean = pd_results_all_contexts.applymap(np.mean).round(round_to)
    pd_results_all_contexts_mean.to_csv(os.path.join(results_dir, 'metrics_mean.csv'), index=True)
    pd_results_all_contexts_mean.to_latex(os.path.join(results_dir, 'metrics_mean.tex'), index=True)
    pd_results_all_contexts_std = pd_results_all_contexts.applymap(np.std).round(round_to)
    pd_results_all_contexts_std.to_csv(os.path.join(results_dir, 'metrics_std.csv'), index=True)
    pd_results_all_contexts_std.to_latex(os.path.join(results_dir, 'metrics_std.tex'), index=True)
    mean_numpy = pd_results_all_contexts_mean.to_numpy()
    std_numpy = pd_results_all_contexts_std.to_numpy()
    # TODO round to
    text = ''
    for mean_row, std_row in zip(mean_numpy, std_numpy):
        for mean, std in zip(mean_row, std_row):
            if math.isnan(mean):
                text += '& '
            else:
                text += f'& {round(mean, round_to)} \\pm {round(std, round_to)} '
        text += '\\\\ \n \\hline \n'

    with open(os.path.join(results_dir, 'metrics_mean_std.tex'), 'w') as f:
        f.write(text)

    print(f"\n------ ALL CONTEXTS METRICS")
    print(pd_results_all_contexts_mean)
    print(pd_results_all_contexts_std)
