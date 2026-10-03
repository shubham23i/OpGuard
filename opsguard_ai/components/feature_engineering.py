from pathlib import Path

import numpy as np

from opsguard_ai.entities.config_entity import FeatureEngineeringConfig
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.logger import logging


class FeatureEngineering:

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    def create_features(self, windows: np.ndarray):
        features = []

        for window in windows:

            mean = np.mean(window, axis=0)
            std = np.std(window, axis=0)
            minimum = np.min(window, axis=0)
            maximum = np.max(window, axis=0)

            first_value = window[0]
            last_value = window[-1]

            change = last_value - first_value

            absolute_change = np.abs(np.diff(window, axis=0))
            volatility = np.mean(absolute_change, axis=0)

            feature_vector = np.concatenate([
                mean,
                std,
                minimum,
                maximum,
                change,
                volatility
            ])

            features.append(feature_vector)

        return np.asarray(features)

    def initiate_feature_engineering(self):

        try:
            input_dir = Path(self.config.input_data_dir)
            output_dir = Path(self.config.output_data_dir)

            train_dir = input_dir / "train"
            test_dir = input_dir / "test"

            output_train_dir = output_dir / "train"
            output_test_dir = output_dir / "test"

            output_train_dir.mkdir(parents=True, exist_ok=True)
            output_test_dir.mkdir(parents=True, exist_ok=True)

            train_files = sorted(train_dir.glob("*.npy"))

            for train_file in train_files:

                machine_name = train_file.stem
                test_file = test_dir / train_file.name

                if not test_file.exists():
                    raise FileNotFoundError(
                        f"Test window file not found for {machine_name}"
                    )

                train_windows = np.load(train_file)
                test_windows = np.load(test_file)

                train_features = self.create_features(train_windows)
                test_features = self.create_features(test_windows)

                np.save(
                    output_train_dir / f"{machine_name}.npy",
                    train_features
                )

                np.save(
                    output_test_dir / f"{machine_name}.npy",
                    test_features
                )

                logging.info(
                    f"Features created for {machine_name} | "
                    f"Train: {train_features.shape} | "
                    f"Test: {test_features.shape}"
                )

            logging.info("Feature engineering completed successfully.")

        except Exception as e:
            raise CustomException(e)