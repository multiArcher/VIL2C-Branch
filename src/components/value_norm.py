"""Running value normalizer used by Yu et al. MAPPO (on-policy ValueNorm)."""

import torch
import torch.nn as nn


class ValueNorm(nn.Module):
    def __init__(self, input_shape, beta=0.99999, epsilon=1e-5):
        super(ValueNorm, self).__init__()
        self.input_shape = (input_shape,) if isinstance(input_shape, int) else tuple(input_shape)
        self.epsilon = epsilon
        self.beta = beta
        self.running_mean = nn.Parameter(torch.zeros(self.input_shape), requires_grad=False)
        self.running_mean_sq = nn.Parameter(torch.zeros(self.input_shape), requires_grad=False)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector):
        if not torch.is_tensor(input_vector):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(self.running_mean.device).reshape(-1, *self.input_shape)
        batch_mean = input_vector.mean(dim=0)
        batch_sq_mean = (input_vector**2).mean(dim=0)
        weight = self.beta
        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector):
        mean, var = self.running_mean_var()
        return (input_vector - mean) / torch.sqrt(var)

    def denormalize(self, input_vector):
        mean, var = self.running_mean_var()
        return input_vector * torch.sqrt(var) + mean
