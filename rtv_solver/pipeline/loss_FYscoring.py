import torch
import torch.nn as nn
from typing import Callable


class _FYGradient(torch.autograd.Function):
    """
    Custom autograd function that injects the Fenchel-Young gradient into the
    computation graph without backpropagating through the ILP oracle.

    Forward value:  ⟨θ, ŷ - y*⟩  (scalar proxy for monitoring)
    Backward grad:  ŷ - y*       (the true FY gradient w.r.t. θ)

    ŷ and y* are treated as constants w.r.t. autograd; their computation
    must happen outside this function under torch.no_grad().
    """

    @staticmethod
    def forward(
        ctx,
        theta: torch.Tensor,        # (n,) — scores with grad
        y_hat: torch.Tensor,        # (n,) — smoothed MAP prediction, detached
        y_star: torch.Tensor,       # (n,) — ground-truth solution, detached
    ) -> torch.Tensor:
        ctx.save_for_backward(y_hat, y_star)
        return torch.dot(theta, y_hat - y_star)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        y_hat, y_star = ctx.saved_tensors
        # gradient w.r.t. theta = ŷ - y*; no gradient for ŷ or y*
        return grad_output * (y_hat - y_star), None, None


class FenchelYoungLoss(nn.Module):
    """
    Fenchel-Young loss for structured prediction via perturb-and-MAP.

    Operates on a single instance at a time: one score vector θ ∈ ℝ^n and one
    ground-truth structured solution y* ∈ {0,1}^n.  The MAP oracle is passed
    at forward() time so the same loss module is reusable across iterations
    whose combinatorial structure (number of trips, vehicles, etc.) changes.

    Loss value:  ⟨θ, ŷ - y*⟩
    Gradient:    ŷ - y*    where  ŷ = E_ε[MAP(θ + σ·ε)],  ε ~ N(0, I)

    The oracle must not track gradients internally; it is called entirely
    under torch.no_grad().
    """

    def __init__(self, num_samples: int = 10, sigma: float = 1.0) -> None:
        """
        Args:
            num_samples: Number of Monte Carlo samples for estimating ŷ.
                         More samples reduce gradient variance at the cost of
                         running the ILP oracle that many times per update.
            sigma:       Standard deviation of the Gaussian perturbation.
                         Larger values increase exploration but smooth the loss
                         more aggressively.
        """
        super().__init__()
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        if sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.num_samples = num_samples
        self.sigma = sigma

    def forward(
        self,
        theta: torch.Tensor,
        y_star: torch.Tensor,
        oracle: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the Fenchel-Young loss for a single instance.

        Args:
            theta:  (n,) — raw scores produced by the neural network.
                    Must carry gradients (requires_grad=True or be part of the
                    computation graph of a model with parameters).
            y_star: (n,) — binary ground-truth solution {0, 1}^n.
                    No gradient required.
            oracle: Callable[[Tensor (n,)], Tensor (n,)]
                    Accepts a (possibly perturbed) score vector and returns a
                    feasible binary assignment.  Must be safe to call under
                    torch.no_grad() and must not track gradients internally.

        Returns:
            Scalar loss tensor.  Call .backward() to propagate gradients to
            the model parameters that produced theta.
        """
        if theta.shape != y_star.shape:
            raise ValueError(
                f"theta and y_star must have the same shape, "
                f"got {theta.shape} vs {y_star.shape}"
            )
        if theta.ndim != 1:
            raise ValueError(
                f"Expected 1-D tensors, got shape {theta.shape}. "
                "Use a single instance; mini-batch FY is not supported."
            )

        # Monte Carlo estimate of ŷ = E_ε[MAP(θ + σε)]
        # All oracle calls run without autograd.
        with torch.no_grad():
            y_samples = []
            for _ in range(self.num_samples):
                noise = torch.randn_like(theta) * self.sigma
                y_k = oracle(theta.detach() + noise)
                y_samples.append(y_k.to(theta.dtype))
            y_hat = torch.stack(y_samples, dim=0).mean(dim=0)   # (n,)

        y_star_d = y_star.detach().to(theta.dtype)
        return _FYGradient.apply(theta, y_hat, y_star_d)
