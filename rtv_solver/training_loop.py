from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.pipeline.model_simpleScoring import ScoringMLP
from rtv_solver.pipeline.feat_builder import FeatureBuilder
from rtv_solver.structure.config import Config
from rtv_solver.visuals.training_loss import plot_loss

from rtv_solver.util.logger import BASIC_LOGGER
import logging
console_logger = logging.getLogger(BASIC_LOGGER)


@dataclass
class TrainingLoopResult:
    updated_driver_runs: list
    epoch_iteration_losses: list[list[Optional[float]]]
    all_iteration_losses: list[Optional[float]]


class COAMLTrainingLoop:
    """
    Minimal training loop that replays the same payload for multiple rounds.

    The same model instance is reused across all epochs and optimized online from the per-iteration Fenchel-Young loss tracked by the pipeline.
    """

    def __init__(
        self,
        config: Config,
        payload: dict,
        model: ScoringMLP | None = None,
    ) -> None:
        self.config = config
        self.payload = payload
        self.model = model if model is not None else ScoringMLP(
            feature_dim=FeatureBuilder.FEATURE_SIZE, hidden_dim=32
        )

    def run(self) -> TrainingLoopResult:
        updated_driver_runs = []
        epoch_iteration_losses: list[list[Optional[float]]] = []
        all_iteration_losses: list[Optional[float]] = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        total_epochs = self.config.EPOCHS
        max_checkpoints = min(10, total_epochs)
        checkpoint_epochs = sorted(
            {
                max(1, round(save_idx * total_epochs / max_checkpoints))
                for save_idx in range(1, max_checkpoints + 1)
            }
        )
        checkpoint_epoch_set = set(checkpoint_epochs)

        for epoch in range(total_epochs):
            epoch_num = epoch + 1
            console_logger.info(
                f"=== COAML training epoch {epoch_num}/{total_epochs} ==="
            )
            # Recreate the pipeline each epoch to keep run-level state isolated,
            # while injecting the same model instance so parameters persist.
            pipeline = COAMLPipeline(self.config, self.payload, model=self.model)
            updated_driver_runs = pipeline.solve_pdptw(
                self.payload,
                mode="train",
                optimizer=optimizer,
                epoch=epoch_num,
            )

            # Keep per-iteration losses from this epoch as produced by the pipeline.
            epoch_losses = list(pipeline.loss_history)
            epoch_iteration_losses.append(epoch_losses)
            all_iteration_losses.extend(epoch_losses)

            # Save a bounded number of evenly spaced checkpoints over full training.
            if epoch_num in checkpoint_epoch_set:
                pipeline.save_model_weights(
                    self.config.OUTPUT_DIR / f"coaml_model_weights_epoch_{epoch_num}.pt"
                )

        # Plot once at the end of the loop (instead of every solve_pdptw call).
        plot_loss(all_iteration_losses, self.config.EPOCHS)
        return TrainingLoopResult(
            updated_driver_runs=updated_driver_runs,
            epoch_iteration_losses=epoch_iteration_losses,
            all_iteration_losses=all_iteration_losses,
        )
