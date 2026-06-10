from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributions as dist
from torch.distributions import Distribution, constraints, kl_divergence
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical

# ========================================
# ============== GUMBEL  =================
# ========================================
class GumbelDistribution(ExpRelaxedCategorical):
    """Gubmel distribution based on ExpRelaxedCategorical distribution."""

    @property
    def probs(self):
        return torch.exp(self.logits).clip(1e-6, 1 - 1e-6)

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        probs = self.probs.clip(1e-6, 1 - 1e-6)
        return OneHotCategorical(probs=probs).sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return torch.exp(super().rsample(sample_shape))

    @property
    def mean(self):
        return self.probs.clip(1e-6, 1 - 1e-6)

    @property
    def mode(self):
        probs = self.probs.clip(1e-6, 1 - 1e-6)
        return OneHotCategorical(probs=probs).mode

    def expand(self, batch_shape, _instance=None):
        return super().expand(batch_shape[:-1], _instance)

    def log_prob(self, value):
        probs = self.probs.clip(1e-6, 1 - 1e-6)
        return OneHotCategorical(probs=probs).log_prob(value)

# @torch.compile
def sample_gumbel(
    shape: Tuple[int, int],
    eps: float = 1e-9,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate a sample from the Gumbel distribution.

    Args:
        shape (Tuple[int, int]): Shape of the sample.
        eps (float, optional): Value to be added to avoid numerical issues. Defaults to 1e-10.

    Returns:
        torch.Tensor: Sample from the Gumbel distribution.
    """
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


# @torch.compile
def gumbel_softmax(
    logits: torch.Tensor,
    shape: Tuple[int, int],
    tau: float = 1.0,
    hard: bool = False,
) -> torch.Tensor:
    """
    Gumbel-Softmax implementation. See https://neptune.ai/blog/gumbel-softmax-loss-function-guide-how-to-implement-it-in-pytorch.
    PyTorchs function caused issues during training.

    Args:
        logits (torch.Tensor): Logits to be used for the Gumbel-Softmax.
        shape (Tuple[int, int]): Shape of the Logits. Required for torchscript
        tau (float, optional): Temperature factor. Defaults to 1.0.
        hard (bool, optional): Hard sampling or soft. Defaults to False.

    Returns:
        torch.Tensor: Sampled categorical distribution.
    """
    if torch.isnan(logits).any():
        raise ValueError("Logits contain NaN values")

    gumbel_noise = sample_gumbel(shape, device=logits.device)
    y = logits + gumbel_noise
    tau = max(tau, 1e-9)
    y_soft = torch.softmax(y / (tau), dim=-1)

    if hard:
        _, ind = y_soft.max(dim=-1)
        y_hard = torch.zeros_like(y_soft).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(shape[0], shape[1])
        y_soft = (y_hard - y_soft).detach() + y_soft

    return y_soft

# ========================================
# ============= CATEGORICAL  =============
# ========================================
class ReparameterizedCategorical(Distribution):
    """
    Reparameterized Categorical Distribution using a Gumbel-Softmax relaxation over logits.
    """
    # @typechecked
    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ):
        """
        Initialize the Reparameterized Categorical Distribution.
        Args:
            logits (Optional[torch.Tensor]): A tensor of logits (unnormalized log probabilities).
            probs (Optional[torch.Tensor]): A tensor of probabilities.
            temperature (float): A temperature parameter for the Gumbel-Softmax distribution.
        """
        self._categorical = torch.distributions.Categorical(
            logits=logits, probs=probs
        )
        self.temperature = temperature

    @property
    def param_shape(self) -> torch.Size:
        """
        Returns the shape of the parameter tensor.
        Returns:
            torch.Size: The shape of the parameter tensor.
        """
        return self._categorical.param_shape

    @property
    def batch_shape(self) -> torch.Size:
        """
        Returns the shape of the batch of distributions.
        Returns:
            torch.Size: The shape of the batch of distributions.
        """
        return self._categorical.batch_shape

    @property
    def event_shape(self) -> torch.Size:
        """
        Returns the shape of the event of the distribution.
        Returns:
            torch.Size: The shape of the event of the distribution.
        """
        return self._categorical.event_shape

    @property
    def support(self) -> torch.Tensor:
        """
        Returns the support of the distribution.
        Returns:
            torch.Tensor: The support of the distribution.
        """
        return self._categorical.support

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Draws a sample from the distribution.

        Args:
            sample_shape (torch.Size, optional): The shape of the sample to draw. Defaults to torch.Size().

        Returns:
            torch.Tensor: The drawn sample.
        """
        return self._categorical.sample(sample_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Reparameterized sampling using Gumbel-Softmax trick.
        """
        if torch.any(torch.isnan(self._categorical.logits)):
            raise Exception("NaN values in logits")
        samples = gumbel_softmax(
            logits=self._categorical.logits,
            shape=tuple(self._categorical.logits.shape),
            tau=self.temperature,
            hard=False,
        )
        # return samples
        return torch.argmax(samples, 1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:

        """
        Calculate the log_prob of a value.

        Args:
            value (torch.Tensor): Input value.

        Returns:
            torch.Tensor: Log probability of the input value.
        """

        return self._categorical.log_prob(value)

    @constraints.dependent_property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        """
        Returns the argument constraints of the distribution.

        Returns:
            Dict[str, Constraint]: Constraint dictionary.
        """
        return self._categorical.arg_constraints

    @property
    def mode(self) -> torch.Tensor:
        """
        Returns the mode of the distribution.
        """
        return self._categorical.mode.unsqueeze(-1)

    def mean(self) -> torch.Tensor:
        """
        Returns the mean of the distribution.
        """
        return self._categorical.mean

# ========================================
# ============ AUX SAMPLINGS  ============
# ========================================
class DistH():
    def __init__(self, config):
        super(DistH).__init__()
        self.prior_mean = config.prior_mean
        self.prior_std = config.prior_std

    def kl_z_prior(self, mean, std):
        p_mean = torch.ones_like(mean).to(mean.device) * self.prior_mean
        p_std = torch.ones_like(mean).to(mean.device) * self.prior_std
        prior_batch = dist.normal.Normal(p_mean, p_std)
        post_batch = dist.normal.Normal(mean, std)
        KL = (kl_divergence(post_batch, prior_batch).sum(1))
        return KL

    def sampling_z(self, mean, std, val=False):
        batch_dist = dist.normal.Normal(mean, std)
        samples = (batch_dist.sample() if val else batch_dist.rsample())
        return samples

    def sampling_q(self, probs, tau, val=False):
        batch_dist = GumbelDistribution(logits=probs, temperature=tau)
        samples = (batch_dist.sample() if val else batch_dist.rsample())
        return samples