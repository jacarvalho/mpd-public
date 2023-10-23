import numpy as np
import torch
from scipy import integrate


def prior_likelihood(z, sigma):
    """The likelihood of a Gaussian distribution with mean zero and
        standard deviation sigma."""
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * torch.log(2 * np.pi * sigma ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * sigma ** 2)


def ode_likelihood(x,
                   score_model,
                   marginal_prob_std,
                   diffusion_coeff,
                   batch_size=64,
                   device='cuda',
                   eps=1e-5):
    """Compute the likelihood with probability flow ODE.

    Args:
      x: Input data.
      score_model: A PyTorch model representing the score-based model.
      marginal_prob_std: A function that gives the standard deviation of the
        perturbation kernel.
      diffusion_coeff: A function that gives the diffusion coefficient of the
        forward SDE.
      batch_size: The batch size. Equals to the leading dimension of `x`.
      device: 'cuda' for evaluation on GPUs, and 'cpu' for evaluation on CPUs.
      eps: A `float` number. The smallest time step for numerical stability.

    Returns:
      z: The latent code for `x`.
      bpd: The log-likelihoods in bits/dim.
    """

    # Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
    epsilon = torch.randn_like(x)

    def divergence_eval(sample, time_steps, epsilon):
        """Compute the divergence of the score-based model with Skilling-Hutchinson."""
        with torch.enable_grad():
            sample.requires_grad_(True)
            score_e = torch.sum(score_model(sample, time_steps) * epsilon)
            grad_score_e = torch.autograd.grad(score_e, sample)[0]
        return torch.sum(grad_score_e * epsilon, dim=(1, 2, 3))

    shape = x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the score-based model for the black-box ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0],))
        with torch.no_grad():
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def divergence_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
        with torch.no_grad():
            # Obtain x(t) by solving the probability flow ODE.
            sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
            time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0],))
            # Compute likelihood.
            div = divergence_eval(sample, time_steps, epsilon)
            return div.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for the black-box solver."""
        time_steps = np.ones((shape[0],)) * t
        sample = x[:-shape[0]]
        logp = x[-shape[0]:]
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        sample_grad = -0.5 * g ** 2 * score_eval_wrapper(sample, time_steps)
        logp_grad = -0.5 * g ** 2 * divergence_eval_wrapper(sample, time_steps)
        return np.concatenate([sample_grad, logp_grad], axis=0)


    init = np.concatenate([x.cpu().numpy().reshape((-1,)), np.zeros((shape[0],))], axis=0)
    # Black-box ODE solver
    res = integrate.solve_ivp(ode_func, (eps, 1.), init, rtol=1e-5, atol=1e-5, method='RK45')
    zp = torch.tensor(res.y[:, -1], device=device)
    z = zp[:-shape[0]].reshape(shape)
    delta_logp = zp[-shape[0]:].reshape(shape[0])
    sigma_max = marginal_prob_std(1.)
    prior_logp = prior_likelihood(z, sigma_max)
    bpd = -(prior_logp + delta_logp) / np.log(2)
    N = np.prod(shape[1:])
    bpd = bpd / N + 8.
    return z, bpd



