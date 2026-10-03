from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from opsguard_ai.entities.config_entity import BaselineDetectionConfig
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.logger import logging


class BaselineDetector:

    def __init__(self, config: BaselineDetectionConfig):
        self.config = config

    def initiate_baseline_detection(self):

        try:
            input_dir = Path(self.config.input_data_dir)
            output_dir = Path(self.config.output_data_dir)
            model_dir = Path(self.config.model_dir)

            train_dir = input_dir / "train"
            test_dir = input_dir / "test"

            output_train_dir = output_dir / "train"
            output_test_dir = output_dir / "test"

            output_train_dir.mkdir(parents=True, exist_ok=True)
            output_test_dir.mkdir(parents=True, exist_ok=True)
            model_dir.mkdir(parents=True, exist_ok=True)

            train_files = sorted(train_dir.glob("*.npy"))

            for train_file in train_files:

                machine_name = train_file.stem
                test_file = test_dir / train_file.name

                if not test_file.exists():
                    raise FileNotFoundError(
                        f"Test feature file not found for {machine_name}"
                    )

                train_features = np.load(train_file)
                test_features = np.load(test_file)

                model = IsolationForest(
                    contamination=self.config.contamination,
                    random_state=self.config.random_state,
                    n_jobs=-1
                )

                model.fit(train_features)

                train_predictions = model.predict(train_features)
                test_predictions = model.predict(test_features)

                train_anomalies = (train_predictions == -1).astype(int)
                test_anomalies = (test_predictions == -1).astype(int)

                train_scores = -model.score_samples(train_features)
                test_scores = -model.score_samples(test_features)

                np.save(
                    output_train_dir / f"{machine_name}.npy",
                    np.column_stack([
                        train_anomalies,
                        train_scores
                    ])
                )

                np.save(
                    output_test_dir / f"{machine_name}.npy",
                    np.column_stack([
                        test_anomalies,
                        test_scores
                    ])
                )

                joblib.dump(
                    model,
                    model_dir / f"{machine_name}.pkl"
                )

                logging.info(
                    f"Baseline detection completed for {machine_name} | "
                    f"Test anomalies: {test_anomalies.sum()}"
                )

            logging.info(
                "Baseline anomaly detection completed successfully."
            )

        except Exception as e:
            raise CustomException(e)