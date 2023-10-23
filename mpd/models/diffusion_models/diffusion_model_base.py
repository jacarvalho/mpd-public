"""
Adapted from https://github.com/jannerm/diffuser
"""
import abc
import time
from collections import namedtuple
from copy import copy

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from abc import ABC

from torch.nn import DataParallel

from mpd.models.diffusion_models.helpers import cosine_beta_schedule, Losses, exponential_beta_schedule
from mpd.models.diffusion_models.sample_functions import extract, apply_hard_conditioning, guide_gradient_steps, \
    ddpm_sample_fn
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import to_numpy


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


def build_context(model, dataset, input_dict):
    # input_dict is already normalized
    context = None
    if model.context_model is not None:
        context = dict()
        # (normalized) features of variable environments
        if dataset.variable_environment:
            env_normalized = input_dict[f'{dataset.field_key_env}_normalized']
            context['env'] = env_normalized

        # tasks
        task_normalized = input_dict[f'{dataset.field_key_task}_normalized']
        context['tasks'] = task_normalized
    return context


class GaussianDiffusionModel(nn.Module, ABC):

    def __init__(self,
                 model=None,
                 variance_schedule='exponential',
                 n_diffusion_steps=100,
                 clip_denoised=True,
                 predict_epsilon=False,
                 loss_type='l2',
                 context_model=None,
                 **kwargs):
        super().__init__()

        self.model = model

        self.context_model = context_model

        self.n_diffusion_steps = n_diffusion_steps

        self.state_dim = self.model.state_dim

        if variance_schedule == 'cosine':
            betas = cosine_beta_schedule(n_diffusion_steps, s=0.008, a_min=0, a_max=0.999)
        elif variance_schedule == 'exponential':
            betas = exponential_beta_schedule(n_diffusion_steps, beta_start=1e-4, beta_end=1.0)
        else:
            raise NotImplementedError

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#
    def predict_noise_from_start(self, x_t, t, x0):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return x0
        else:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
            ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, hard_conds, context, t):
        if context is not None:
            context = self.context_model(context)

        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, context))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, hard_conds, context=None, return_chain=False,
                      sample_fn=ddpm_sample_fn,
                      n_diffusion_steps_without_noise=0,
                      **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_hard_conditioning(x, hard_conds)

        chain = [x] if return_chain else None

        for i in reversed(range(-n_diffusion_steps_without_noise, self.n_diffusion_steps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, hard_conds, context, t, **sample_kwargs)
            x = apply_hard_conditioning(x, hard_conds)

            if return_chain:
                chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)
            return x, chain

        return x

    @torch.no_grad()
    def ddim_sample(
        self, shape, hard_conds,
        context=None, return_chain=False,
        t_start_guide=torch.inf,
        guide=None,
        n_guide_steps=1,
        **sample_kwargs,
    ):
        # Adapted from https://github.com/ezhang7423/language-control-diffusion/blob/63cdafb63d166221549968c662562753f6ac5394/src/lcd/models/diffusion.py#L226
        device = self.betas.device
        batch_size = shape[0]
        total_timesteps = self.n_diffusion_steps
        sampling_timesteps = self.n_diffusion_steps // 5
        eta = 0.

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(0, total_timesteps - 1, steps=sampling_timesteps + 1, device=device)
        times = torch.cat((torch.tensor([-1], device=device), times))
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device=device)
        x = apply_hard_conditioning(x, hard_conds)

        chain = [x] if return_chain else None

        for time, time_next in time_pairs:
            t = make_timesteps(batch_size, time, device)
            t_next = make_timesteps(batch_size, time_next, device)

            model_out = self.model(x, t, context)

            x_start = self.predict_start_from_noise(x, t=t, noise=model_out)
            pred_noise = self.predict_noise_from_start(x, t=t, x0=model_out)

            if time_next < 0:
                x = x_start
                x = apply_hard_conditioning(x, hard_conds)
                if return_chain:
                    chain.append(x)
                break

            alpha = extract(self.alphas_cumprod, t, x.shape)
            alpha_next = extract(self.alphas_cumprod, t_next, x.shape)

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            x = x_start * alpha_next.sqrt() + c * pred_noise

            # guide gradient steps before adding noise
            if guide is not None:
                if torch.all(t_next < t_start_guide):
                    x = guide_gradient_steps(
                        x,
                        hard_conds=hard_conds,
                        guide=guide,
                        **sample_kwargs
                    )

            # add noise
            noise = torch.randn_like(x)
            x = x + sigma * noise
            x = apply_hard_conditioning(x, hard_conds)

            if return_chain:
                chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)
            return x, chain

        return x

    @torch.no_grad()
    def conditional_sample(self, hard_conds, horizon=None, batch_size=1, ddim=False, **sample_kwargs):
        '''
            hard conditions : hard_conds : { (time, state), ... }
        '''
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.state_dim)

        if ddim:
            return self.ddim_sample(shape, hard_conds, **sample_kwargs)

        return self.p_sample_loop(shape, hard_conds, **sample_kwargs)

    def forward(self, cond, *args, **kwargs):
        raise NotImplementedError
        return self.conditional_sample(cond, *args, **kwargs)

    @torch.no_grad()
    def warmup(self, horizon=64, device='cuda'):
        shape = (2, horizon, self.state_dim)
        x = torch.randn(shape, device=device)
        t = make_timesteps(2, 1, device)
        self.model(x, t, context=None)

    @torch.no_grad()
    def run_inference(self, context=None, hard_conds=None, n_samples=1, return_chain=False, **diffusion_kwargs):
        # context and hard_conds must be normalized
        hard_conds = copy(hard_conds)
        context = copy(context)

        # repeat hard conditions and contexts for n_samples
        for k, v in hard_conds.items():
            new_state = einops.repeat(v, 'd -> b d', b=n_samples)
            hard_conds[k] = new_state

        if context is not None:
            for k, v in context.items():
                context[k] = einops.repeat(v, 'd -> b d', b=n_samples)

        # Sample from diffusion model
        samples, chain = self.conditional_sample(
            hard_conds, context=context, batch_size=n_samples, return_chain=True, **diffusion_kwargs
        )

        # chain: [ n_samples x (n_diffusion_steps + 1) x horizon x (state_dim)]
        # extract normalized trajectories
        trajs_chain_normalized = chain

        # trajs: [ (n_diffusion_steps + 1) x n_samples x horizon x state_dim ]
        trajs_chain_normalized = einops.rearrange(trajs_chain_normalized, 'b diffsteps h d -> diffsteps b h d')

        if return_chain:
            return trajs_chain_normalized

        # return the last denoising step
        return trajs_chain_normalized[-1]

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, context, t, hard_conds):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_hard_conditioning(x_noisy, hard_conds)

        # context model
        if context is not None:
            context = self.context_model(context)

        # diffusion model
        x_recon = self.model(x_noisy, t, context)
        x_recon = apply_hard_conditioning(x_recon, hard_conds)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, context, *args):
        batch_size = x.shape[0]
        t = torch.randint(0, self.n_diffusion_steps, (batch_size,), device=x.device).long()
        return self.p_losses(x, context, t, *args)

