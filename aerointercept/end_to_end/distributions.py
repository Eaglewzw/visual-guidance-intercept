"""Numerically stable bounded Gaussian policy helpers."""
import math

import torch
import torch.nn.functional as F


ACTION_EPSILON = 1e-6


def squashed_normal_sample(latent_mean: torch.Tensor,
                           log_std: torch.Tensor):
    """Reparameterized tanh-Gaussian sample and corrected log probability."""
    std = log_std.exp().expand_as(latent_mean)
    distribution = torch.distributions.Normal(latent_mean, std)
    latent = distribution.rsample()
    action = torch.tanh(latent)
    log_jacobian = 2.0 * (
        math.log(2.0) - latent - F.softplus(-2.0 * latent))
    log_probability = (
        distribution.log_prob(latent) - log_jacobian).sum(dim=-1)
    return action, log_probability


def squashed_normal_log_probability(action: torch.Tensor,
                                     latent_mean: torch.Tensor,
                                     log_std: torch.Tensor):
    """Log probability of an already bounded action under a tanh Gaussian."""
    bounded = action.clamp(-1.0 + ACTION_EPSILON, 1.0 - ACTION_EPSILON)
    latent = torch.atanh(bounded)
    std = log_std.exp().expand_as(latent_mean)
    distribution = torch.distributions.Normal(latent_mean, std)
    log_jacobian = torch.log(1.0 - bounded.square() + ACTION_EPSILON)
    return (distribution.log_prob(latent) - log_jacobian).sum(dim=-1)
