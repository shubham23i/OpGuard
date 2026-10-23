
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
from opsguard_ai.entities.config_entity import DataPreprocessingConfig
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.logger import logging


class DataPreprocessing:

    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def initiate_data_preprocessing(self):
        try:
            input_dir = Path(self.config.input_data_dir)
            train_dir = input_dir / "train"
            test_dir = input_dir / "test"

            output_dir = Path(self.config.processed_data_dir)
            output_train_dir = output_dir / "train"
            output_test_dir = output_dir / "test"

            output_train_dir.mkdir(parents=True, exist_ok=True)
            output_test_dir.mkdir(parents=True, exist_ok=True)

            scaler_dir = Path(self.config.scaler_dir)
            scaler_dir.mkdir(parents=True, exist_ok=True)

            train_files = sorted(train_dir.glob("*.txt"))

            if not train_files:
                raise FileNotFoundError(
                    f"No training files found in {train_dir}"
                )

            min_scale = 0.01

            for train_file in train_files:
                machine_name = train_file.stem
                test_file = test_dir / train_file.name

                if not test_file.exists():
                    raise FileNotFoundError(
                        f"Test file not found for {machine_name}"
                    )

                train_data = pd.read_csv(train_file, header=None)
                test_data = pd.read_csv(test_file, header=None)

                if train_data.shape[1] != test_data.shape[1]:
                    raise ValueError(
                        f"Feature count mismatch for {machine_name}"
                    )

                if not np.isfinite(train_data.to_numpy()).all():
                    raise ValueError(
                        f"Non-finite training values in {machine_name}"
                    )

                if not np.isfinite(test_data.to_numpy()).all():
                    raise ValueError(
                        f"Non-finite test values in {machine_name}"
                    )

                scaler = StandardScaler()
                train_scaled = scaler.fit_transform(train_data)
                test_scaled = scaler.transform(test_data)

                small_scale = scaler.scale_ < min_scale
                scaler.scale_[small_scale] = min_scale

                train_scaled = (
                    train_data.to_numpy() - scaler.mean_
                ) / scaler.scale_

                test_scaled = (
                    test_data.to_numpy() - scaler.mean_
                ) / scaler.scale_

                pd.DataFrame(train_scaled).to_csv(
                    output_train_dir / train_file.name,
                    header=False,
                    index=False,
                )

                pd.DataFrame(test_scaled).to_csv(
                    output_test_dir / test_file.name,
                    header=False,
                    index=False,
                )

                joblib.dump(
                    scaler,
                    scaler_dir / f"{machine_name}.pkl",
                )

                logging.info(
                    f"Preprocessed {machine_name}; "
                    f"protected {int(small_scale.sum())} low-scale features"
                )

            logging.info("Data preprocessing completed successfully.")

        except Exception as e:
            raise CustomException(e,sys)
