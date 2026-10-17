
import json

import numpy as np
import torch
import sys
from opsguard_ai.components.anomaly_detector import GRUAutoencoder
from opsguard_ai.entities.config_entity import ExplainabilityConfig
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.logger import logging


class Explainability:
    def __init__(self, config: ExplainabilityConfig):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _load_predictions(self, prediction_file):
        predictions = np.load(prediction_file, allow_pickle=False)

        if predictions.ndim == 1:
            labels = predictions
        elif predictions.ndim == 2 and predictions.shape[1] >= 1:
            labels = predictions[:, 0]
        else:
            raise ValueError(
                f"Unsupported prediction shape: {predictions.shape}"
            )

        return labels.astype(int).reshape(-1)

    def _load_model(self, model_file, input_size):
        model = GRUAutoencoder(
            input_size=input_size,
            hidden_size=64,
            latent_size=16,
            num_layers=1,
            dropout=0.0,
        ).to(self.device)

        state_dict = torch.load(
            model_file,
            map_location=self.device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.eval()

        return model

    def _calculate_feature_errors(self, windows, model):
        batches = []

        with torch.no_grad():
            for start in range(0, len(windows), self.config.batch_size):
                batch_array = windows[
                    start:start + self.config.batch_size
                ]

                batch = torch.as_tensor(
                    batch_array,
                    dtype=torch.float32,
                    device=self.device,
                )

                reconstruction = model(batch)

                if reconstruction.shape != batch.shape:
                    raise ValueError(
                        "Model reconstruction shape does not match "
                        "the input window shape."
                    )

                errors = torch.mean(
                    (batch - reconstruction) ** 2,
                    dim=1,
                )

                batches.append(errors.cpu().numpy())

        if not batches:
            return np.empty((0, windows.shape[-1]), dtype=np.float32)

        return np.concatenate(batches, axis=0)

    def initiate_explainability(self):
        try:
            logging.info("Starting Explainability")

            window_dir = self.config.input_window_dir / "test"
            score_dir = self.config.input_score_dir / "test"
            prediction_dir = self.config.input_prediction_dir
            output_dir = self.config.output_dir

            output_dir.mkdir(parents=True, exist_ok=True)

            if not window_dir.exists():
                raise FileNotFoundError(
                    f"Test window directory not found: {window_dir}"
                )

            window_files = sorted(window_dir.glob("*.npy"))

            if not window_files:
                raise FileNotFoundError(
                    f"No window files found in {window_dir}"
                )

            generated = 0
            skipped = []

            for window_file in window_files:
                machine = window_file.stem

                score_file = score_dir / f"{machine}.npy"
                prediction_file = prediction_dir / f"{machine}.npy"
                model_file = self.config.model_dir / f"{machine}.pt"

                required_files = [
                    score_file,
                    prediction_file,
                    model_file,
                ]

                missing = [
                    str(path)
                    for path in required_files
                    if not path.exists()
                ]

                if missing:
                    logging.warning(
                        "Skipping %s; missing artifacts: %s",
                        machine,
                        missing,
                    )
                    skipped.append({
                        "machine": machine,
                        "missing_files": missing,
                    })
                    continue

                windows = np.load(
                    window_file,
                    allow_pickle=False,
                )

                if windows.ndim != 3:
                    raise ValueError(
                        f"{machine}: expected 3D windows, "
                        f"received shape {windows.shape}"
                    )

                if len(windows) == 0:
                    logging.warning("Skipping empty windows for %s", machine)
                    continue

                scores = np.asarray(
                    np.load(score_file, allow_pickle=False)
                ).reshape(-1)

                labels = self._load_predictions(prediction_file)

                if not (
                    len(windows) == len(scores) == len(labels)
                ):
                    raise ValueError(
                        f"{machine}: length mismatch: "
                        f"windows={len(windows)}, "
                        f"scores={len(scores)}, labels={len(labels)}"
                    )

                model = self._load_model(
                    model_file,
                    input_size=windows.shape[-1],
                )

                feature_errors = self._calculate_feature_errors(
                    windows,
                    model,
                )

                feature_names = [
                    f"metric_{index:02d}"
                    for index in range(feature_errors.shape[1])
                ]

                anomaly_indices = np.flatnonzero(labels == 1)
                reports = []

                for index in anomaly_indices:
                    ranked_indices = np.argsort(
                        feature_errors[index]
                    )[::-1][:self.config.top_k]

                    reports.append({
                        "window_index": int(index),
                        "anomaly_score": float(scores[index]),
                        "top_contributing_metrics": [
                            {
                                "metric": feature_names[metric_index],
                                "reconstruction_error": float(
                                    feature_errors[index, metric_index]
                                ),
                            }
                            for metric_index in ranked_indices
                        ],
                    })

                report = {
                    "machine": machine,
                    "method": "GRU reconstruction error",
                    "feature_count": int(windows.shape[-1]),
                    "total_windows": int(len(windows)),
                    "anomalous_windows": int(len(anomaly_indices)),
                    "explained_windows": len(reports),
                    "interpretation": (
                        "Metrics are ranked by reconstruction error. "
                        "They are statistical contributors, not confirmed "
                        "causes of an anomaly."
                    ),
                    "reports": reports,
                }

                output_file = output_dir / f"{machine}.json"

                with open(
                    output_file,
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(report, file, indent=4)

                generated += 1

            summary = {
                "machines_found": len(window_files),
                "reports_generated": generated,
                "machines_skipped": skipped,
                "method": "GRU reconstruction error",
            }

            with open(
                output_dir / "summary.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(summary, file, indent=4)

            logging.info(
                "Explainability completed: %s reports generated",
                generated,
            )

            return summary

        except Exception as e:
            logging.exception("Explainability failed")
            raise CustomException(e,sys)
