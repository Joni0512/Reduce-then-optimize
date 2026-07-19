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

# TODO simplest solution to run training, should probably be improved in the future and more flexible for other instances etc.
TRAINING_FILES = [
    "lc101", "lc102", "lc103", "lc104", "lc105", "lc106", "lc107",
    # "lc201", "lc202", "lc203", "lc204", "lc205", "lc206",
    "lr101", "lr102", "lr103", "lr104", "lr105", "lr106", "lr107", "lr108", "lr109", "lr110",
    # "lr201", "lr202", "lr203", "lr204", "lr205", "lr206", "lr207", "lr208", "lr209",
    "lrc101", "lrc102", "lrc103", "lrc104", "lrc105", "lrc106",
    # "lrc201", "lrc202", "lrc203", "lrc204", "lrc205", "lrc206"
]
VALIDATION_FILES = [
    "lc108", "lc109",
    # "lc207", "lc208",
    "lr111", "lr112",
    # "lr210", "lr211",
    "lrc107", "lrc108",
    #"lrc207", "lrc208"
]

# NOTE DEBUG MODE
# TRAINING_FILES = ["lc101", "lc105"]
# VALIDATION_FILES = ["lc108", "lc201"]

SUPPORTED_INPUT_EXTENSIONS = {".json", ".pkl", ".txt"}


def _split_train_val_files(
    dir_path: Path, train_stems: set[str], val_stems: set[str]
) -> tuple[list[Path], list[Path]]:
    """
    Split files in dir_path into train/val by stem using the given stem sets.
    All files must be in INPUT_DIR; the sets define which stems go to train vs val.
    """
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {dir_path}")
    all_files = sorted(
        p for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS
    )
    train_files = [p for p in all_files if p.stem in train_stems]
    val_files = [p for p in all_files if p.stem in val_stems]
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
    COAML training loop. Three modes:
    - Single payload: replay same payload for multiple epochs (no train/val split).
    - Train/val payloads: train on `payload`, evaluate (mode="eval") on `val_payload` each
      epoch, keep the best-val-service-rate checkpoint. For datasets like NYC that are one
      big request pool rather than many Li&Lim-style instance files.
    - Train/val dirs: train on TRAINING_FILES, validate on VALIDATION_FILES (both from INPUT_DIR).
      One model, never reset. Validation results stored per file for later analysis.
    """
    def __init__(
        self,
        config: Config,
        payload: dict | None = None,
        model: ScoringMLP | None = None,
        input_path: Path | None = None,
        extra_validation_files: list[str] | None = None,
        extra_training_files: list[str] | None = None,
        val_payload: dict | None = None,
        val_input_path: Path | None = None,
    ) -> None:
        self.config = config
        self.payload = payload
        self.model = model or ScoringMLP(
            feature_dim=FeatureBuilder.FEATURE_SIZE, hidden_dim=config.HIDDEN_DIM
        )
        # TODO load model from existing file to improve learning
        self.input_path = input_path
        self.val_payload = val_payload
        self.val_input_path = val_input_path
        # 2026-07-19: additive, opt-in extension of TRAINING_FILES/VALIDATION_FILES
        # for testing on instance classes the standard 23/6-file split never covers
        # (LC2/LR2/LRC2 - see conversation "welche Instanzen ... Request pruner").
        # Motivation for extra_training_files specifically: the request/pair
        # pruners both train on a mixed-class split (MODEL_CONFIGS["mixed_all"],
        # 36 instances incl. 17 class-2 train files), but SIL's own scoring MLP
        # previously trained on class-1-only TRAINING_FILES - so any class-2
        # damage measured via extra_validation_files alone was confounded with
        # plain domain shift (SIL never having seen class-2 patterns at all),
        # not attributable to the pruner specifically. Passing the pruner's own
        # 17 class-2 train instances here removes that confound.
        self.train_stems = set(TRAINING_FILES) | set(extra_training_files or [])
        self.val_stems = set(VALIDATION_FILES) | set(extra_validation_files or [])

    def run(self) -> TrainingLoopResult:
        if self.config.INPUT_DIR:
            return self._run_train_val_dirs()
        if self.payload is None:
            raise ValueError("payload required when INPUT_DIR is not set")
        if self.val_payload is not None:
            return self._run_train_val_payloads()
        return self._run_single_payload()

    def _run_train_val_dirs(self) -> TrainingLoopResult:
        """
        Train on TRAINING_FILES, validate on VALIDATION_FILES (both from INPUT_DIR).
        Single model shared across all training; validation results stored per file.
        """
        root = Path(__file__).resolve().parent.parent
        data_dir = root / self.config.INPUT_DIR
        train_files, val_files = _split_train_val_files(
            data_dir, self.train_stems, self.val_stems
        )
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


        #validate files to see what rolling horizon can even do on the optimal path
        for v_path in val_files:
                v_cleared = _load_and_clear_payload(v_path)
                out_dir = self.config.OUTPUT_DIR / "optimal_val" / v_path.stem
                out_dir.mkdir(parents=True, exist_ok=True)
                file_cfg = replace(self.config, OUTPUT_DIR=out_dir)
                pipeline = COAMLPipeline(
                    file_cfg, v_cleared, model=self.model, imitation_solution_path=v_path,
                )
                driver_runs = pipeline.solve_pdptw(v_cleared, mode="optimal")
                _save_validation_results(file_cfg, v_cleared, driver_runs)

        for epoch in range(self.config.EPOCHS):
            epoch_num = epoch + 1
            console_logger.info(f"=== Epoch {epoch_num}/{self.config.EPOCHS} ===")

            # Train on all training files; model is never reset
            for path in train_files:
                cleared = _load_and_clear_payload(path)
                pipeline = COAMLPipeline(
                    self.config, cleared, model=self.model, epoch=epoch_num,
                    imitation_solution_path=path,
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

            # Validate: each epoch with the current model to run model on val files after complete training, store results per file
            for v_path in val_files:
                v_cleared = _load_and_clear_payload(v_path)
                out_dir = self.config.OUTPUT_DIR / "val" / f"epoch_{epoch}" /v_path.stem
                out_dir.mkdir(parents=True, exist_ok=True)
                file_cfg = replace(self.config, OUTPUT_DIR=out_dir)
                pipeline = COAMLPipeline(
                    file_cfg, v_cleared, model=self.model, epoch=epoch_num,
                    imitation_solution_path=v_path,
                )
                driver_runs = pipeline.solve_pdptw(v_cleared, mode="eval")
                _save_validation_results(file_cfg, v_cleared, driver_runs)
        # save losses before it runs the further validation (derisking getting cut off from the results)
        save_json(losses_per_file, self.config.OUTPUT_DIR / "training_loss_per_file.json")
        
        
        # Validate training files after complete training, store results per file per epoch
        # runnning this takes too much time and does not provide as much value except overfitting results --> we are not going to run this for all epoch but at the end to see whether it as least can fit its trainind data
        for t_path in train_files:
            t_cleared = _load_and_clear_payload(t_path)
            out_dir = self.config.OUTPUT_DIR / "train_val" / f"epoch_{epoch}" / t_path.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            file_cfg = replace(self.config, OUTPUT_DIR=out_dir)
            pipeline = COAMLPipeline(
                file_cfg, t_cleared, model=self.model, epoch=epoch_num,
                imitation_solution_path=t_path,
            )
            driver_runs = pipeline.solve_pdptw(t_cleared, mode="eval")
            _save_validation_results(file_cfg, t_cleared, driver_runs)

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

    def _run_train_val_payloads(self) -> TrainingLoopResult:
        """
        Train on self.payload across self.config.EPOCHS, evaluating on self.val_payload
        each epoch via mode="eval" (the model's own learned scores, no gradient update -
        NOT mode="train"'s expert-imitation replay). Keeps the checkpoint with the best
        val service rate (serviced / total_requests) as coaml_model_weights_best_val.pt.

        2026-07-19: added because the only other single-payload path (_run_single_payload)
        trains AND is scored on the exact same payload every epoch, always via mode="train"
        - fine for Li&Lim's --input_dir case (many small instances, a real train/val file
        split), but for NYC's one-big-pool single-manifest runs it meant every past "SIL"
        result actually reported a freshly-initialized, undertrained model's expert-replay
        score on its own training data, never the model's own policy on unseen data.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.LEARNING_RATE
        )
        all_iteration_losses: list[Optional[float]] = []
        epoch_iteration_losses: list[list[Optional[float]]] = []
        updated_driver_runs: list = []
        best_val_service_rate = -1.0
        best_epoch = -1

        for epoch in range(self.config.EPOCHS):
            epoch_num = epoch + 1
            console_logger.info(f"=== COAML train/val epoch {epoch_num}/{self.config.EPOCHS} ===")

            pipeline = COAMLPipeline(
                self.config, self.payload, model=self.model, epoch=epoch_num,
                imitation_solution_path=self.input_path,
            )
            updated_driver_runs = pipeline.solve_pdptw(
                self.payload, mode="train", optimizer=optimizer
            )
            epoch_losses = list(pipeline.loss_history)
            epoch_iteration_losses.append(epoch_losses)
            all_iteration_losses.extend(epoch_losses)

            val_out_dir = self.config.OUTPUT_DIR / "val" / f"epoch_{epoch_num}"
            val_out_dir.mkdir(parents=True, exist_ok=True)
            val_cfg = replace(self.config, OUTPUT_DIR=val_out_dir)
            val_pipeline = COAMLPipeline(
                val_cfg, self.val_payload, model=self.model, epoch=epoch_num,
                imitation_solution_path=self.val_input_path,
            )
            val_driver_runs = val_pipeline.solve_pdptw(self.val_payload, mode="eval")

            val_stats_payload = {
                PayloadKeys.DEPOT: self.val_payload[PayloadKeys.DEPOT],
                PayloadKeys.REQUESTS: self.val_payload[PayloadKeys.REQUESTS],
                PayloadKeys.DRIVERS: val_driver_runs,
                PayloadKeys.TIME_MATRIX: self.val_payload.get(PayloadKeys.TIME_MATRIX),
            }
            evaluator = StatsParser(val_cfg, payload=val_stats_payload)
            _, val_stats, val_violations = evaluator.evaluate(val_stats_payload)
            evaluator.add_total_time(total_time=0.0)
            save_json(val_stats_payload, val_cfg.OUTPUT_DIR / "result_driver_runs.json")
            save_json(
                {"stats": val_stats, "violations": val_violations},
                val_cfg.OUTPUT_DIR / "results.json",
            )

            val_service_rate = val_stats.serviced / max(val_stats.total_requests, 1)
            console_logger.info(
                f"Epoch {epoch_num}: val service rate = {val_service_rate:.4f} "
                f"({val_stats.serviced}/{val_stats.total_requests})"
            )

            torch.save(
                {"model_state_dict": self.model.state_dict()},
                self.config.OUTPUT_DIR / f"coaml_model_weights_epoch_{epoch_num}.pt",
            )
            if val_service_rate > best_val_service_rate:
                best_val_service_rate = val_service_rate
                best_epoch = epoch_num
                torch.save(
                    {"model_state_dict": self.model.state_dict()},
                    self.config.OUTPUT_DIR / "coaml_model_weights_best_val.pt",
                )

        console_logger.info(
            f"Best val service rate = {best_val_service_rate:.4f} at epoch {best_epoch} "
            f"-> {self.config.OUTPUT_DIR / 'coaml_model_weights_best_val.pt'}"
        )
        plot_loss(all_iteration_losses, self.config.EPOCHS, self.config.OUTPUT_DIR)
        return TrainingLoopResult(
            updated_driver_runs=updated_driver_runs,
            epoch_iteration_losses=epoch_iteration_losses,
            all_iteration_losses=all_iteration_losses,
        )

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
                self.config, self.payload, model=self.model, epoch=epoch_num,
                imitation_solution_path=self.input_path,
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
