from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from opsguard_ai.components.anomaly_detector import GRUAutoencoder
from opsguard_ai.entities.config_entity import RootCauseConfig
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.logger import logging


class RootCauseAnalyzer:

    def __init__(self, config: RootCauseConfig):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def load_model(self, model_path: Path):
        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True
        )

        model = GRUAutoencoder(
            input_size=38,
            hidden_size=64,
            latent_size=16,
            num_layers=1,
            dropout=0.0
        ).to(self.device)

        model.load_state_dict(checkpoint)
        model.eval()

        return model

    def calculate_metric_errors(self, model, windows):
        loader = DataLoader(
            TensorDataset(
                torch.tensor(windows, dtype=torch.float32)
            ),
            batch_size=self.config.batch_size,
            shuffle=False
        )

        errors = []

        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                reconstructed = model(batch)

                metric_errors = torch.mean(
                    (reconstructed - batch) ** 2,
                    dim=1
                )

                errors.append(metric_errors.cpu().numpy())

        return np.concatenate(errors, axis=0)

    def analyze_event(
        self,
        metric_errors,
        event,
        threshold
    ):
        start = event["start_window"]
        end = event["end_window"]

        event_errors = metric_errors[start:end + 1]

        mean_errors = np.mean(event_errors, axis=0)
        peak_errors = np.max(event_errors, axis=0)

        order = np.argsort(mean_errors)[::-1]
        candidates = []

        for index in order[:self.config.top_k]:
            candidates.append({
                "metric_id": int(index),
                "mean_reconstruction_error": round(
                    float(mean_errors[index]), 6
                ),
                "peak_reconstruction_error": round(
                    float(peak_errors[index]), 6
                ),
                "relative_contribution_pct": round(
                    float(
                        100 * mean_errors[index] /
                        (mean_errors.sum() + 1e-12)
                    ),
                    2
                )
            })

        return {
            "start_window": start,
            "end_window": end,
            "duration_windows": event["duration_windows"],
            "root_cause_candidates": candidates
        }

    def initiate_root_cause_analysis(self):
        try:
            window_dir = Path(self.config.input_window_dir) / "test"
            score_dir = Path(self.config.input_score_dir)
            event_dir = Path(self.config.input_event_dir)
            model_dir = Path(self.config.model_dir)
            output_dir = Path(self.config.output_dir)

            output_dir.mkdir(parents=True, exist_ok=True)

            event_files = sorted(event_dir.glob("*.json"))

            for event_file in event_files:
                machine_name = event_file.stem

                window_file = window_dir / f"{machine_name}.npy"
                score_file = score_dir / f"{machine_name}.npy"
                model_file = model_dir / f"{machine_name}.pt"

                for path in [window_file, score_file, model_file]:
                    if not path.exists():
                        raise FileNotFoundError(
                            f"Required artifact missing: {path}"
                        )

                windows = np.load(window_file).astype(np.float32)
                scores = np.load(score_file)

                with open(event_file, "r", encoding="utf-8") as file:
                    event_data = json.load(file)

                model = self.load_model(model_file)
                metric_errors = self.calculate_metric_errors(
                    model, windows
                )

                if len(metric_errors) != len(scores):
                    raise ValueError(
                        f"Window/score mismatch for {machine_name}"
                    )

                candidates = [
                    self.analyze_event(
                        metric_errors,
                        event,
                        threshold=None
                    )
                    for event in event_data["events"]
                ]

                output = {
                    "machine_id": machine_name,
                    "analysis_type": "reconstruction_error_attribution",
                    "interpretation": (
                        "Ranked telemetry metrics are diagnostic "
                        "candidates, not confirmed root causes."
                    ),
                    "total_events_analyzed": len(candidates),
                    "events": candidates
                }

                with open(
                    output_dir / f"{machine_name}.json",
                    "w",
                    encoding="utf-8"
                ) as file:
                    json.dump(output, file, indent=4)

                logging.info(
                    f"Root-cause candidate analysis completed for "
                    f"{machine_name} | Events: {len(candidates)}"
                )

            logging.info(
                "Root-cause candidate analysis completed successfully."
            )

        except Exception as e:
            raise CustomException(e)