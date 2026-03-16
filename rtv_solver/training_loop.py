from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional
import torch

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.stats_parser import StatsParser
from rtv_solver.pipeline.model_simpleScoring import ScoringMLP
from rtv_solver.pipeline.feat_builder import FeatureBuilder
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import save_json
from rtv_solver.visuals.training_loss import plot_loss, plot_loss_per_file

from rtv_solver.util.logger import BASIC_LOGGER
import logging

console_logger = logging.getLogger(BASIC_LOGGER)

# TODO simplest solution to run training, should probably be improved in the future
# TRAINING_FILES = [
#     "lc101", "lc102", "lc103", "lc104", "lc105", "lc106", "lc107",
#     "lc201", "lc202", "lc203", "lc204", "lc205", "lc206",
#     "lr101", "lr102", "lr103", "lr104", "lr105", "lr106", "lr107", "lr108", "lr109", "lr110",
#     "lr201", "lr202", "lr203", "lr204", "lr205", "lr206", "lr207", "lr208", "lr209",
#     "lrc101", "lrc102", "lrc103", "lrc104", "lrc105", "lrc106",
#     "lrc201", "lrc202", "lrc203", "lrc204", "lrc205", "lrc206"
# ]
# VALIDATION_FILES = [
#     "lc108", "lc109",
#     "lc207", "lc208",
#     "lr111", "lr112",
#     "lr210", "lr211",
#     "lrc107", "lrc108",
#     "lrc207", "lrc208"
# ]

# NOTE DEBUG MODE
TRAINING_FILES = ["lc101", "lc105"]
VALIDATION_FILES = ["lc108", "lc201"]

SUPPORTED_INPUT_EXTENSIONS = {".json", ".pkl", ".txt"}
TRAIN_STEMS = set(TRAINING_FILES)
VAL_STEMS = set(VALIDATION_FILES)


def _split_train_val_files(dir_path: Path) -> tuple[list[Path], list[Path]]:
    """
    Split files in dir_path into train/val by stem using TRAINING_FILES and VALIDATION_FILES.
    All files must be in INPUT_DIR; lists define which stems go to train vs val.
    """
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {dir_path}")
    all_files = sorted(
        p for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS
    )
    train_files = [p for p in all_files if p.stem in TRAIN_STEMS]
    val_files = [p for p in all_files if p.stem in VAL_STEMS]
    return train_files, val_files


def _load_and_clear_payload(input_path: Path) -> dict:
    """Load payload from file and clear vehicle manifests for solver."""
    data = PayloadParser.load_input_data(input_path)
    return PayloadParser.clear_vehicle_manifests(data)


def _save_validation_results(cfg: Config, payload: dict, driver_runs: list) -> None:
    """Parse manifests with StatsParser and save result_driver_runs.json + results.json."""
    stats_payload = {
        PayloadKeys.DEPOT: payload[PayloadKeys.DEPOT],
        PayloadKeys.REQUESTS: payload[PayloadKeys.REQUESTS],
        PayloadKeys.DRIVERS: driver_runs,
        PayloadKeys.TIME_MATRIX: payload.get(PayloadKeys.TIME_MATRIX),
    }
    evaluator = StatsParser(cfg, payload=stats_payload)
    feasible, stats, violations = evaluator.evaluate(stats_payload)
    evaluator.add_total_time(total_time=0.0)
    save_json(stats_payload, cfg.OUTPUT_DIR / "result_driver_runs.json")
    save_json({"stats": stats, "violations": violations}, cfg.OUTPUT_DIR / "results.json")


@dataclass
class TrainingLoopResult:
    updated_driver_runs: list
    epoch_iteration_losses: list[list[Optional[float]]]
    all_iteration_losses: list[Optional[float]]


class COAMLTrainingLoop:
    """
    COAML training loop. Two modes:
    - Single payload: replay same payload for multiple epochs (no train/val split).
    - Train/val dirs: train on TRAINING_FILES, validate on VALIDATION_FILES (both from INPUT_DIR).
      One model, never reset. Validation results stored per file for later analysis.
    """

    def __init__(
        self,
        config: Config,
        payload: dict | None = None,
        model: ScoringMLP | None = None,
    ) -> None:
        self.config = config
        self.payload = payload
        self.model = model or ScoringMLP(
            feature_dim=FeatureBuilder.FEATURE_SIZE, hidden_dim=32
        )

    def run(self) -> TrainingLoopResult:
        if self.config.INPUT_DIR:
            return self._run_train_val_dirs()
        if self.payload is None:
            raise ValueError("payload required when INPUT_DIR is not set")
        return self._run_single_payload()

    def _run_train_val_dirs(self) -> TrainingLoopResult:
        """
        Train on TRAINING_FILES, validate on VALIDATION_FILES (both from INPUT_DIR).
        Single model shared across all training; validation results stored per file.
        """
        root = Path(__file__).resolve().parent.parent
        data_dir = root / self.config.INPUT_DIR
        train_files, val_files = _split_train_val_files(data_dir)
        if not train_files:
            raise ValueError(f"No training files in {data_dir}")

        console_logger.info(
            f"Train: {len(train_files)} files, Val: {len(val_files)} files"
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.LEARNING_RATE
        )
        all_losses: list[Optional[float]] = []
        losses_per_file: dict[str, list[Optional[float]]] = {}
        last_driver_runs: list = []

        for epoch in range(self.config.EPOCHS):
            epoch_num = epoch + 1
            console_logger.info(f"=== Epoch {epoch_num}/{self.config.EPOCHS} ===")

            # Train on all training files; model is never reset
            for path in train_files:
                cleared = _load_and_clear_payload(path)
                pipeline = COAMLPipeline(
                    self.config, cleared, model=self.model, epoch=epoch_num
                )
                last_driver_runs = pipeline.solve_pdptw(
                    cleared, mode="train", optimizer=optimizer
                )
                all_losses.extend(pipeline.loss_history)
                losses_per_file.setdefault(path.stem, []).extend(pipeline.loss_history)

            # Save epoch checkpoints for resumability
            if epoch_num in self._checkpoint_epochs():
                torch.save(
                    {"model_state_dict": self.model.state_dict()},
                    self.config.OUTPUT_DIR / f"coaml_model_weights_epoch_{epoch_num}.pt",
                )

        # Validate: run model on val files after complete training, store results per file
        for v_path in val_files:
            v_cleared = _load_and_clear_payload(v_path)
            out_dir = self.config.OUTPUT_DIR / "val" / v_path.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            file_cfg = replace(self.config, OUTPUT_DIR=out_dir)
            pipeline = COAMLPipeline(
                file_cfg, v_cleared, model=self.model, epoch=epoch_num
            )
            driver_runs = pipeline.solve_pdptw(v_cleared, mode="eval")
            _save_validation_results(file_cfg, v_cleared, driver_runs)

        # Validate training files after complete training, store results per file per epoch
        for t_path in train_files:
            t_cleared = _load_and_clear_payload(t_path)
            out_dir = self.config.OUTPUT_DIR / "train_val" / t_path.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            file_cfg = replace(self.config, OUTPUT_DIR=out_dir)
            pipeline = COAMLPipeline(
                file_cfg, t_cleared, model=self.model, epoch=epoch_num
            )
            driver_runs = pipeline.solve_pdptw(t_cleared, mode="eval")
            _save_validation_results(file_cfg, t_cleared, driver_runs)

        plot_loss_per_file(losses_per_file, self.config.OUTPUT_DIR)
        return TrainingLoopResult(
            updated_driver_runs=last_driver_runs,
            epoch_iteration_losses=[],
            all_iteration_losses=all_losses,
        )

    def _checkpoint_epochs(self) -> set[int]:
        """Epochs at which to save model checkpoints (evenly spaced)."""
        total = self.config.EPOCHS
        max_ckpts = min(10, total)
        return {
            max(1, round(i * total / max_ckpts))
            for i in range(1, max_ckpts + 1)
        }

    def _run_single_payload(self) -> TrainingLoopResult:
        """Replay same payload for multiple epochs (original behavior)."""
        updated_driver_runs = []
        epoch_iteration_losses: list[list[Optional[float]]] = []
        all_iteration_losses: list[Optional[float]] = []
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.LEARNING_RATE
        )
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
            pipeline = COAMLPipeline(
                self.config, self.payload, model=self.model, epoch=epoch_num
            )
            updated_driver_runs = pipeline.solve_pdptw(
                self.payload, mode="train", optimizer=optimizer
            )
            epoch_losses = list(pipeline.loss_history)
            epoch_iteration_losses.append(epoch_losses)
            all_iteration_losses.extend(epoch_losses)
            if epoch_num in checkpoint_epoch_set:
                pipeline.save_model_weights(
                    self.config.OUTPUT_DIR
                    / f"coaml_model_weights_epoch_{epoch_num}.pt"
                )

        plot_loss(all_iteration_losses, self.config.EPOCHS, self.config.OUTPUT_DIR)
        return TrainingLoopResult(
            updated_driver_runs=updated_driver_runs,
            epoch_iteration_losses=epoch_iteration_losses,
            all_iteration_losses=all_iteration_losses,
        )
