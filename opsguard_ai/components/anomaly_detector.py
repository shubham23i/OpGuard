from pathlib import Path
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from opsguard_ai.entities.config_entity import AnomalyDetectionConfig
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.logger import logging


class GRUAutoencoder(nn.Module):

    def __init__(
        self,
        input_size=38,
        hidden_size=64,
        latent_size=16,
        num_layers=1,
        dropout=0.0
    ):
        super().__init__()

        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)

        self.decoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        _, hidden = self.encoder(x)

        latent = self.to_latent(hidden[-1])
        decoder_input = self.from_latent(latent).unsqueeze(1)

        decoder_input = decoder_input.repeat(
            1, x.size(1), 1
        )

        decoded, _ = self.decoder(decoder_input)

        return self.output_layer(decoded)


class AnomalyDetector:

    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def set_seed(self):
        random.seed(self.config.random_state)
        np.random.seed(self.config.random_state)
        torch.manual_seed(self.config.random_state)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_state)

    def create_model(self):
        return GRUAutoencoder(
            input_size=38,
            hidden_size=self.config.hidden_size,
            latent_size=self.config.latent_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(self.device)

    def train_model(self, model, train_data):
        validation_size = max(1, int(len(train_data) * 0.1))
        split_index = len(train_data) - validation_size

        if split_index < 1:
            raise ValueError("Not enough training windows.")

        training_data = train_data[:split_index]
        validation_data = train_data[split_index:]

        train_tensor = torch.tensor(
            training_data, dtype=torch.float32
        )
        validation_tensor = torch.tensor(
            validation_data, dtype=torch.float32
        )

        train_loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=self.config.batch_size,
            shuffle=True
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate
        )

        criterion = nn.MSELoss()
        best_loss = float("inf")
        best_weights = None
        patience_counter = 0

        for epoch in range(self.config.epochs):
            model.train()
            total_train_loss = 0.0

            for (batch,) in train_loader:
                batch = batch.to(self.device)

                optimizer.zero_grad()
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * len(batch)

            train_loss = total_train_loss / len(training_data)

            model.eval()
            with torch.no_grad():
                validation_tensor_device = validation_tensor.to(self.device)
                reconstructed = model(validation_tensor_device)
                validation_loss = criterion(
                    reconstructed,
                    validation_tensor_device
                ).item()

            logging.info(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Validation Loss: {validation_loss:.6f}"
            )

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                logging.info("Early stopping triggered.")
                break

        if best_weights is not None:
            model.load_state_dict(best_weights)

        return model

    def calculate_scores(self, model, windows):
        model.eval()
        scores = []

        dataset = TensorDataset(
            torch.tensor(windows, dtype=torch.float32)
        )

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                reconstructed = model(batch)

                errors = torch.mean(
                    (reconstructed - batch) ** 2,
                    dim=(1, 2)
                )

                scores.extend(errors.cpu().numpy())

        return np.asarray(scores)

    def initiate_anomaly_detection(self):
        try:
            self.set_seed()

            input_dir = Path(self.config.input_data_dir)
            model_dir = Path(self.config.model_dir)
            score_dir = Path(self.config.score_dir)

            train_dir = input_dir / "train"
            test_dir = input_dir / "test"

            train_score_dir = score_dir / "train"
            test_score_dir = score_dir / "test"

            train_score_dir.mkdir(parents=True, exist_ok=True)
            test_score_dir.mkdir(parents=True, exist_ok=True)
            model_dir.mkdir(parents=True, exist_ok=True)

            train_files = sorted(train_dir.glob("*.npy"))

            for train_file in train_files:
                machine_name = train_file.stem
                test_file = test_dir / train_file.name

                if not test_file.exists():
                    raise FileNotFoundError(
                        f"Test windows not found for {machine_name}"
                    )

                train_windows = np.load(train_file).astype(np.float32)
                test_windows = np.load(test_file).astype(np.float32)

                model = self.create_model()
                model = self.train_model(model, train_windows)

                train_scores = self.calculate_scores(
                    model, train_windows
                )
                test_scores = self.calculate_scores(
                    model, test_windows
                )

                np.save(
                    train_score_dir / f"{machine_name}.npy",
                    train_scores
                )
                np.save(
                    test_score_dir / f"{machine_name}.npy",
                    test_scores
                )

                torch.save(
                    model.state_dict(),
                    model_dir / f"{machine_name}.pt"
                )

                logging.info(
                    f"GRU Autoencoder completed for {machine_name} | "
                    f"Test windows: {len(test_scores)}"
                )

            logging.info("GRU Autoencoder training completed.")

        except Exception as e:
            raise CustomException(e)