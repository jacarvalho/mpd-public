import abc
import time

import einops
import numpy as np
import torch
from torch import nn

from mp_baselines.planners.costs.cost_functions import CostGPTrajectory
from mp_baselines.planners.costs.factors.mp_priors_multi import MultiMPPrior
from torch_robotics.torch_planning_objectives.fields.distance_fields import interpolate_points_v1
from torch_robotics.torch_utils.torch_utils import to_torch


class GuideManagerTrajectories(nn.Module):

    def __init__(self, dataset, cost, clip_grad=False, clip_grad_rule='norm', max_grad_norm=1., max_grad_value=0.1,
                 interpolate_trajectories_for_collision=False,
                 num_interpolated_points_for_collision=128,
                 use_velocity_from_finite_difference=False,
                 start_state_pos=None,
                 goal_state_pos=None,
                 num_steps=100,
                 robot=None,
                 n_samples=1,
                 tensor_args=None,
                 **kwargs):
        super().__init__()
        self.cost = cost
        self.dataset = dataset

        self.interpolate_trajectories_for_collision = interpolate_trajectories_for_collision
        self.num_interpolated_points_for_collision = num_interpolated_points_for_collision

        self.clip_grad = clip_grad
        self.clip_grad_rule = clip_grad_rule
        self.max_grad_norm = max_grad_norm
        self.max_grad_value = max_grad_value

        # velocity
        self.use_velocity_from_finite_difference = use_velocity_from_finite_difference
        self.robot = robot
        self.start_state_pos = start_state_pos
        self.goal_state_pos = goal_state_pos
        # initialize velocity trajectory with a constant velocity
        self.velocity = self.robot.get_velocity(
            MultiMPPrior.const_vel_trajectory(
                start_state_pos,
                goal_state_pos,
                robot.dt,
                num_steps,
                self.robot.q_dim,
                set_initial_final_vel_to_zero=True,
                tensor_args=tensor_args)
        )
        self.velocity = to_torch(self.velocity, **tensor_args)
        self.velocity = einops.repeat(self.velocity, "H D -> B H D", B=n_samples)

    def forward(self, x_pos_normalized):
        x_pos = x_pos_normalized.clone()
        with torch.enable_grad():
            x_pos.requires_grad_(True)
            self.velocity.requires_grad_(True)

            # unnormalize x
            # x is normalized, but the guides are defined on unnormalized trajectory space
            x_pos = self.dataset.unnormalize_trajectories(x_pos)

            if self.interpolate_trajectories_for_collision:
                # finer interpolation of trajectory for better collision avoidance
                x_interpolated = interpolate_points_v1(x_pos, num_interpolated_points=self.num_interpolated_points_for_collision)
            else:
                x_interpolated = x_pos

            # compute costs
            # append the current velocity trajectory to the position trajectory only for non-interpolated trajectories
            if self.use_velocity_from_finite_difference:
                x_vel = self.robot.get_velocity(x_pos)
                x_pos_vel = torch.cat((x_pos, x_vel), dim=-1)
            else:
                x_pos_vel = torch.cat((x_pos, self.velocity), dim=-1)

            cost_l, weight_grad_cost_l = self.cost(x_pos_vel, x_interpolated=x_interpolated, return_invidual_costs_and_weights=True)
            grad = 0
            grad_velocity = 0
            for cost, weight_grad_cost in zip(cost_l, weight_grad_cost_l):
                if torch.is_tensor(cost):
                    # y.sum() is a surrogate to compute gradients of independent quantities over the batch dimension
                    # x are the support points. Compute gradients wrt x, not x_interpolated
                    if self.use_velocity_from_finite_difference:
                        grad_cost = torch.autograd.grad([cost.sum()], [x_pos], retain_graph=True)[0]
                    else:
                        grad_cost, grad_cost_velocity = torch.autograd.grad([cost.sum()], [x_pos, self.velocity])

                    # clip gradients
                    grad_cost_clipped = self.clip_gradient(grad_cost)
                    if not self.use_velocity_from_finite_difference:
                        grad_cost_velocity_clipped = self.clip_gradient(grad_cost_velocity)

                    # zeroing gradients at start and goal
                    grad_cost_clipped[..., 0, :] = 0.
                    grad_cost_clipped[..., -1, :] = 0.
                    if not self.use_velocity_from_finite_difference:
                        grad_cost_velocity_clipped[..., 0, :] = 0.
                        grad_cost_velocity_clipped[..., -1, :] = 0.

                    # combine gradients
                    grad_cost_clipped_weighted = weight_grad_cost * grad_cost_clipped
                    grad += grad_cost_clipped_weighted

                    if not self.use_velocity_from_finite_difference:
                        grad_cost_velocity_clipped_weighted = weight_grad_cost * grad_cost_velocity_clipped
                        grad_velocity += grad_cost_velocity_clipped_weighted

            # Update the velocity
            if not self.use_velocity_from_finite_difference:
                self.velocity = self.velocity - grad_velocity

        # gradient ascent
        grad = -1. * grad
        return grad

    def clip_gradient(self, grad):
        if self.clip_grad:
            if self.clip_grad_rule == 'norm':
                return self.clip_grad_by_norm(grad)
            elif self.clip_grad_rule == 'value':
                return self.clip_grad_by_value(grad)
            else:
                raise NotImplementedError
        else:
            return grad

    def clip_grad_by_norm(self, grad):
        # clip gradient by norm
        if self.clip_grad:
            grad_norm = torch.linalg.norm(grad + 1e-6, dim=-1, keepdims=True)
            scale_ratio = torch.clip(grad_norm, 0., self.max_grad_norm) / grad_norm
            grad = scale_ratio * grad
        return grad

    def clip_grad_by_value(self, grad):
        # clip gradient by value
        if self.clip_grad:
            grad = torch.clip(grad, -self.max_grad_value, self.max_grad_value)
        return grad


class GuideManagerTrajectoriesWithVelocity(nn.Module):

    def __init__(self, dataset, cost, clip_grad=False, clip_grad_rule='norm', max_grad_norm=1., max_grad_value=0.1,
                 interpolate_trajectories_for_collision=False,
                 num_interpolated_points_for_collision=128,
                 start_state_pos=None,
                 goal_state_pos=None,
                 num_steps=100,
                 robot=None,
                 n_samples=1,
                 tensor_args=None,
                 **kwargs):
        super().__init__()
        self.cost = cost
        self.dataset = dataset

        self.interpolate_trajectories_for_collision = interpolate_trajectories_for_collision
        self.num_interpolated_points_for_collision = num_interpolated_points_for_collision

        self.clip_grad = clip_grad
        self.clip_grad_rule = clip_grad_rule
        self.max_grad_norm = max_grad_norm
        self.max_grad_value = max_grad_value

    def forward(self, x_normalized):
        x = x_normalized.clone()
        with torch.enable_grad():
            x.requires_grad_(True)

            # unnormalize x
            # x is normalized, but the guides are defined on unnormalized trajectory space
            x = self.dataset.unnormalize_trajectories(x)

            if self.interpolate_trajectories_for_collision:
                # finer interpolation of trajectory for better collision avoidance
                x_interpolated = interpolate_points_v1(x, num_interpolated_points=self.num_interpolated_points_for_collision)
            else:
                x_interpolated = x

            # compute costs
            # append the current velocity trajectory to the position trajectory only for non-interpolated trajectories
            cost_l, weight_grad_cost_l = self.cost(x, x_interpolated=x_interpolated, return_invidual_costs_and_weights=True)
            grad = 0
            for cost, weight_grad_cost in zip(cost_l, weight_grad_cost_l):
                if torch.is_tensor(cost):
                    # y.sum() is a surrogate to compute gradients of independent quantities over the batch dimension
                    # x are the support points. Compute gradients wrt x, not x_interpolated
                    grad_cost = torch.autograd.grad([cost.sum()], [x], retain_graph=True)[0]

                    # clip gradients
                    grad_cost_clipped = self.clip_gradient(grad_cost)

                    # zeroing gradients at start and goal
                    grad_cost_clipped[..., 0, :] = 0.
                    grad_cost_clipped[..., -1, :] = 0.

                    # combine gradients
                    grad_cost_clipped_weighted = weight_grad_cost * grad_cost_clipped
                    grad += grad_cost_clipped_weighted

        # gradient ascent
        grad = -1. * grad
        return grad

    def clip_gradient(self, grad):
        if self.clip_grad:
            if self.clip_grad_rule == 'norm':
                return self.clip_grad_by_norm(grad)
            elif self.clip_grad_rule == 'value':
                return self.clip_grad_by_value(grad)
            else:
                raise NotImplementedError
        else:
            return grad

    def clip_grad_by_norm(self, grad):
        # clip gradient by norm
        if self.clip_grad:
            grad_norm = torch.linalg.norm(grad + 1e-6, dim=-1, keepdims=True)
            scale_ratio = torch.clip(grad_norm, 0., self.max_grad_norm) / grad_norm
            grad = scale_ratio * grad
        return grad

    def clip_grad_by_value(self, grad):
        # clip gradient by value
        if self.clip_grad:
            grad = torch.clip(grad, -self.max_grad_value, self.max_grad_value)
        return grad




class GuideBase(nn.Module, abc.ABC):
    def __init__(self, scale=1e-3, tensor_args=None, **kwargs):
        super().__init__()
        self.tensor_args = tensor_args
        self.scale = scale

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def gradients(self, x):
        x.requires_grad_()
        y = self(x)
        # y.sum() is a surrogate to compute gradients of independent quantities over the batch dimension
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()
        return y, grad


class GuideCollisionAvoidance(GuideBase):
    """
    Computes the collision sdf for all points in a trajectory
    """
    def __init__(self, env, **kwargs):
        super().__init__(**kwargs)
        self.env = env

    def forward(self, x):
        collision_cost = self.env.compute_collision_cost(x, field_type='sdf')
        cost = collision_cost.sum(-1)
        return -1 * cost  # maximize


class GuideSmoothnessFiniteDifferenceVelocity(GuideBase):
    """
    Smoothness cost as the central finite difference of velocity, aka acceleration
    """
    def __init__(self, env, method='central', **kwargs):
        super().__init__(**kwargs)
        self.env = env
        self.method = method

    def forward(self, x):
        if self.method == 'central':
            vel = self.env.get_q_velocity(x)
            acc = 0.5 * (vel[..., 1:, :] - vel[..., :-1, :])
            # minimize sum of accelerations along trajectory
            cost = torch.linalg.norm(acc, dim=-1).sum(-1)
        else:
            raise NotImplementedError
        return -1 * cost  # maximize


class GuideSmoothnessGPPrior(GuideBase):
    """
    Smoothness cost as a GP Prior
    """
    def __init__(self, n_dofs, n_support_points_des, start_state, dt, cost_sigmas, **kwargs):
        super().__init__(**kwargs)
        # self.cost_prior = CostGP(n_dofs, n_support_points_des, start_state, dt, cost_sigmas, **kwargs)
        self.cost_prior = CostGPTrajectory(n_dofs, n_support_points_des, start_state, dt, cost_sigmas, **kwargs)

    def forward(self, x):
        cost = self.cost_prior(x)
        return -1 * cost  # maximize


class GuideStateGoal(GuideBase):
    """
    State desired cost
    """
    def __init__(self, state_des, idx, **kwargs):
        super().__init__(**kwargs)
        self.state_des = state_des
        self.idx = idx

    def forward(self, x):
        assert x.ndim >= 3
        cost = torch.linalg.norm(x[..., self.idx, :] - self.state_des, dim=-1)
        return -1 * cost  # maximize


class GuideTrajectorySmoothnessFirstOrder(GuideBase):
    """
    Computes the first-order smoothness of a trajectory
    d(q_0,...,q_{H-1}) = sum_{i=1}^{H-1}(||q_i-q_{i-1}||^2)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        # x: [batch, horizon, dim]
        position_diff = torch.diff(x, dim=1)
        distance = torch.linalg.norm(position_diff, dim=-1).sum(-1)
        return -1. * distance  # maximize the negative distance = minimize distance


class GuideSE3OrientationGoal(GuideBase):
    """
    End effector orientation goal
    """
    def __init__(self, env, rot_des=None, **kwargs):
        super().__init__(**kwargs)
        self.env = env
        if rot_des is not None:
            self.rot_des = rot_des.to(**self.tensor_args)
        else:
            self.rot_des = torch.eye(1).to(**self.tensor_args)

        self.w_rot = 1.

    def forward(self, q):
        b = 1
        h = 1
        if q.ndim == 1:
            q = q.unsqueeze(0).unsqueeze(0)
        elif q.ndim == 2:
            b = q.shape[0]
            q = q.unsqueeze(0)
        elif q.ndim == 3:
            b = q.shape[0]
            h = q.shape[1]
        elif q.ndim > 3:
            raise NotImplementedError

        # batch, trajectory length, q dimension
        q = einops.rearrange(q, 'b h d -> (b h) d')
        # link_tensor = self.diff_panda.compute_forward_kinematics_all_links(q)
        link_tensor_EE = self.env.diff_panda.compute_forward_kinematics_link_list(q, link_list=[self.env.link_name_ee])
        # reshape to batch, trajectory, link poses
        link_tensor_EE = einops.rearrange(link_tensor_EE, '(b h) 1 d1 d2 -> b h d1 d2', b=b, h=h)

        H_des = link_tensor_EE.clone()
        H_des[..., :3, :3] = self.rot_des

        cost = SE3_distance(link_tensor_EE, H_des, w_rot=self.w_rot)
        cost = cost.sum(-1)
        return -1 * cost  # maximize




class GuideTrajectoryLastPoint(GuideBase):
    """
    https://arxiv.org/pdf/2301.06015.pdf
    """
    def __init__(self, goal_state, **kwargs):
        super().__init__(**kwargs)
        self._goal_state = goal_state

    def forward(self, x):
        # x: [batch, horizon, dim]
        l1_dist_goal_state = torch.abs(self._goal_state - x).sum(-1)
        # Equation A5
        # objective = torch.exp(1/l1_dist_goal_state).sum(-1)
        # Equation A8
        objective = -1. * l1_dist_goal_state.sum(-1)
        return objective


class GuideTrajectoryObstacleAvoidanceMultiSphere(GuideBase):
    """
    Computes the sdf for all points in a trajectory
    """
    def __init__(self, obstacles, tensor_args=None, **kwargs):
        super().__init__(**kwargs)
        obst_params = dict(obst_type='sdf')
        self.shape = MultiSphere(tensor_args, **obst_params)
        self.convert_obstacles_to_shape(obstacles)
        self._max_sdf = 0.02

    def convert_obstacles_to_shape(self, obstacles):
        centers = []
        radii = []
        for obstacle in obstacles:
            assert isinstance(obstacle, ObstacleSphere), "Only accepts circles for now"
            centers.append(obstacle.get_center())
            radii.append(obstacle.radius)
        centers = np.array(centers)
        radii = np.array(radii)
        self.shape.set_obst(centers=centers, radii=radii)

    def forward(self, x):
        sdf_points = self.shape.compute_cost(x)
        cost_points = torch.relu(self._max_sdf - sdf_points)
        # cost_points = smooth_distance_penalty(sdf_points)
        cost_trajectory = cost_points.sum(-1)
        return -1 * cost_trajectory  # maximize




def smooth_distance_penalty(d, max_sdf=0.1):
    # TODO this does not allow the gradient to backpropagate because of the torch.where operator
    raise NotImplementedError
    # https://www.ri.cmu.edu/pub_files/2009/5/icra09-chomp.pdf
    # Figure 2
    d_new = torch.zeros_like(d)
    idxs1 = torch.where(d < 0)
    d[idxs1] = -d[idxs1] + 0.5*max_sdf
    idxs2 = torch.where(torch.logical_and(d >= 0, d <= max_sdf))
    d[idxs2] = 1/(2*max_sdf)*(d[idxs2] - max_sdf)**2
    return d_new


class GuideStochGPMP(GuideBase):

    def __init__(self, env, start_state, goal_state, tensor_args, **kwargs):
        super().__init__(**kwargs)

        self.env = env
        self.start_state = start_state
        self.goal_state = goal_state

        self.tensor_args = tensor_args

    def forward(self, x):
        pass

    def gradients(self, x):
        # One step of local StochGPMP
        y, grad = self.plan_sgpmp(x)
        return y, grad

    def plan_sgpmp(self, prior_traj, sgpmp_opt_iters=1, step_size=0.05, return_trajs=False):
        prior_traj = to_torch(prior_traj, **self.tensor_args)

        b, h, d = prior_traj.shape
        n_support_points_des = h

        # SGPMP planner
        n_dofs = self.env.n_dofs
        # Add velocities to state
        start_state = torch.cat((self.start_state, torch.zeros(n_dofs, **self.tensor_args)))
        multi_goal_states = einops.repeat(torch.cat((self.goal_state, torch.zeros(n_dofs, **self.tensor_args))),
                                          'd -> b d', b=b)

        dt = 0.001 * 64 / n_support_points_des
        num_particles_per_goal = 1
        num_samples = 50
        sgpmp_opt_iters = sgpmp_opt_iters

        # Construct cost functions
        cost_sigmas = dict(
            sigma_start=0.0001,
            sigma_gp=0.3,
        )
        cost_prior = CostGP(
            n_dofs, n_support_points_des, start_state, dt,
            cost_sigmas, self.tensor_args
        )

        sigma_goal_prior = 0.0001
        cost_goal_prior = CostGoalPrior(
            n_dofs,
            n_support_points_des,
            multi_goal_states=multi_goal_states,
            num_particles_per_goal=num_particles_per_goal,
            num_samples=num_samples,
            sigma_goal_prior=sigma_goal_prior,
            tensor_args=self.tensor_args
        )

        sigma_coll = 1e-5
        cost_obst_2D = CostCollision(
            n_dofs, n_support_points_des,
            field=self.env.obstacle_map,
            sigma_coll=sigma_coll,
            tensor_args=self.tensor_args
        )
        cost_func_list = [cost_prior, cost_goal_prior, cost_obst_2D]
        cost_composite = CostComposite(n_dofs, n_support_points_des, cost_func_list)

        # Prior mean trajectory
        initial_particle_means = None
        if prior_traj is not None:
            # Interpolate and smooth to desired trajectory length
            # prior_traj = smoothen_trajectory(prior_traj, n_support_points_des, self.tensor_args, smooth_factor=0.01)
            # Reshape for sgpmp interface
            prior_traj = einops.repeat(prior_traj, 'b h d -> b n h d', n=num_particles_per_goal)
            # Add velocities to state
            initial_particle_means = torch.cat((prior_traj, torch.zeros_like(prior_traj)), dim=-1)

        sgpmp_params = dict(
            start_state=start_state,
            multi_goal_states=multi_goal_states,
            cost=cost_composite,
            initial_particle_means=initial_particle_means,
            num_particles_per_goal=num_particles_per_goal,
            num_samples=num_samples,
            n_support_points=n_support_points_des,
            dt=dt,
            n_dof=n_dofs,
            opt_iters=sgpmp_opt_iters,
            temp=0.1,
            step_size=step_size,
            sigma_start_init=1e-3,
            sigma_goal_init=1e-3,
            sigma_gp_init=5.,
            sigma_start_sample=1e-3,
            sigma_goal_sample=1e-3,
            sigma_gp_sample=5.,
            tensor_args=self.tensor_args,
        )

        sgpmp_planner = StochGPMP(**sgpmp_params)
        obs = {}

        # Optimize
        sgpmp_time_start = time.time()
        _, _, _, _, costs, grad = sgpmp_planner.optimize(**obs)

        # costs of all particles
        costs = costs.sum(1)

        # gradient includes velocities
        grad = grad[..., :2]

        sgpmp_time_finish = time.time()

        controls, _, trajectories, trajectory_means, weights = sgpmp_planner.get_recent_samples()

        # print(f'SGPMP time: {sgpmp_time_finish - sgpmp_time_start:.4f} sec')

        if return_trajs:
            return trajectories

        return costs, grad
