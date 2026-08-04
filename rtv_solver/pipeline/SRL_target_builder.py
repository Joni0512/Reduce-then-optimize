import torch


class SRLTargetBuilder:
    """
    Builds a soft target signal for the Fenchel-Young loss without
    requiring a known optimal solution. See docs/SRL_Design.md, Phase 1.
    """

    def __init__(self, num_samples: int = 10, sigma_b: float = 1.0) -> None:
        self.num_samples = num_samples  # m: number of perturbations
        self.sigma_b = sigma_b          # standard deviation of the perturbation

    def perturb_scores(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Draws self.num_samples independent Gaussian perturbations of theta.

        theta: (n,) raw scores from ScoringMLP (one scalar per trip
               candidate + one scalar per reject action).

        Returns: (num_samples, n) tensor of perturbed score vectors
                 eta_k = theta + sigma_b * eps_k, eps_k ~ N(0, I).
        """
        theta_detached = theta.detach()
        noise = torch.randn(
            self.num_samples, *theta_detached.shape,
            device=theta_detached.device, dtype=theta_detached.dtype,
        )
        return theta_detached.unsqueeze(0) + self.sigma_b * noise