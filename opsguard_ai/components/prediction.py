
import json
import sys
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from opsguard_ai.components.anomaly_detector import GRUAutoencoder
from opsguard_ai.utils.logger import logging
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.common import create_directories


class PredictionPipeline:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _load_thresholds(self):
        thresholds = np.load(
            self.config.threshold_file,
            allow_pickle=True,
        )

        if thresholds.shape == ():
            thresholds = thresholds.item()

        if isinstance(thresholds, dict):
            return thresholds

        raise ValueError(
            "Expected thresholds.npy to contain a machine-to-threshold "
            "dictionary. Adapt this loader to the actual saved format."
        )

    def _load_machine_data(self, path):
        data = np.loadtxt(path, delimiter=",", dtype=np.float32)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if not np.isfinite(data).all():
            raise ValueError(f"Invalid numeric values in {path}")

        return data

    def predict_machine(self, machine_name, split="test"):
        try:
            data_path = (
                self.config.input_data_dir
                / split
                / f"{machine_name}.txt"
            )
            model_path = (
                self.config.model_dir
                / f"{machine_name}.pt"
            )

            if not data_path.exists():
                raise FileNotFoundError(
                    f"Machine data not found: {data_path}"
                )

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Trained model not found: {model_path}"
                )

            thresholds = self._load_thresholds()

            if machine_name not in thresholds:
                raise KeyError(
                    f"No threshold registered for {machine_name}"
                )

            threshold = float(thresholds[machine_name])
            data = self._load_machine_data(data_path)

            if data.shape[1] != 38:
                raise ValueError(
                    f"Expected 38 features, got {data.shape[1]}"
                )

            if len(data) < self.config.window_size:
                raise ValueError(
                    f"Need at least {self.config.window_size} rows "
                    f"to create a window."
                )

            windows = np.stack([
                data[i:i + self.config.window_size]
                for i in range(
                    len(data) - self.config.window_size + 1
                )
            ])

            model = GRUAutoencoder(
                input_size=38,
                hidden_size=self.config.hidden_size,
                latent_size=self.config.latent_size,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
            ).to(self.device)

            model.load_state_dict(
                torch.load(
                    model_path,
                    map_location=self.device,
                    weights_only=True,
                )
            )
            model.eval()

            loader = DataLoader(
                TensorDataset(torch.from_numpy(windows)),
                batch_size=self.config.batch_size,
                shuffle=False,
            )

            scores = []

            with torch.no_grad():
                for (batch,) in loader:
                    batch = batch.to(self.device)
                    reconstructed = model(batch)
                    batch_scores = torch.mean(
                        (reconstructed - batch) ** 2,
                        dim=(1, 2),
                    )
                    scores.extend(
                        batch_scores.cpu().numpy().tolist()
                    )

            scores = np.asarray(scores)
            predictions = (scores > threshold).astype(int)

            result = {
                "machine": machine_name,
                "split": split,
                "threshold": threshold,
                "window_count": int(len(scores)),
                "anomaly_count": int(predictions.sum()),
                "anomaly_fraction": float(predictions.mean()),
                "predictions": predictions.tolist(),
                "scores": scores.tolist(),
            }

            create_directories([self.config.output_dir])

            output_path = (
                self.config.output_dir
                / f"{machine_name}_{split}.json"
            )

            with open(
                output_path, "w", encoding="utf-8"
            ) as file:
                json.dump(result, file, indent=4)

            logging.info(
                f"Prediction completed for {machine_name}: "
                f"{predictions.sum()} anomalous windows"
            )

            return result

        except Exception as e:
            raise CustomException(e,sys)
