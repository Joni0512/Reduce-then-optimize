"""
2026-10-12: stand-alone critic TD update step for the "buffered_joint" actor/critic scheme (see chat, srl_training_loop.py).
The ORIGINAL inline critic update in COAMLPipeline.solve_pdptw (coaml_pipeline.py, "for _ in range(num_updates)" block) stays untouched and
is still used by the "per_window" and "buffered" modes; this function re-implements ONE iteration of that loop for entries sampled from the
shared joint buffer, with identical maths: live critic prediction per entry, TD target r + gamma * target critic(s') (min over both target critics
when twin, via td_target_for_step), MSE averaged over the batch, clip_grad_norm_(1.0), Adam step for critic and (when present) critic2 on the same
targets. Equivalence with the inline loop is checked in tests/test_critic_update.py.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from rtv_solver.pipeline.td_target_builder import td_target_for_step


def critic_td_update_step(
    entries: list,
    critic: torch.nn.Module,
    critic_optimizer: torch.optim.Optimizer,
    target_critic: torch.nn.Module,
    gamma: float,
    critic2: torch.nn.Module | None = None,
    critic_optimizer2: torch.optim.Optimizer | None = None,
    target_critic2: torch.nn.Module | None = None,
    max_norm: float = 1.0,  # 2026-10-15: gradient-clipping threshold (default = original 1.0)
) -> float:
    """One critic update on `entries` (ReplayEntry list, td_bootstrap mode). Returns the batch loss of critic 1."""
    batch_q_preds = [
        critic(entry.step.request_features, entry.step.vehicle_features, entry.step.edge_index)
        for entry in entries
    ]
    batch_targets = [
        torch.tensor(
            td_target_for_step(entry.reward, entry.next_step, target_critic, gamma, target_critic2=target_critic2),
            dtype=torch.float32,
        )
        for entry in entries
    ]
    batch_loss = torch.stack([F.mse_loss(q, t) for q, t in zip(batch_q_preds, batch_targets)]).mean()
    critic_optimizer.zero_grad(set_to_none=True)
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=max_norm)
    critic_optimizer.step()

    if critic2 is not None:
        batch_q_preds2 = [
            critic2(entry.step.request_features, entry.step.vehicle_features, entry.step.edge_index)
            for entry in entries
        ]
        batch_loss2 = torch.stack([F.mse_loss(q, t) for q, t in zip(batch_q_preds2, batch_targets)]).mean()
        critic_optimizer2.zero_grad(set_to_none=True)
        batch_loss2.backward()
        torch.nn.utils.clip_grad_norm_(critic2.parameters(), max_norm=max_norm)
        critic_optimizer2.step()
    return batch_loss.item()
