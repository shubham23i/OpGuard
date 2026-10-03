from pathlib import Path

import numpy as np

from opsguard_ai.entities.config_entity import ThresholdManagerConfig
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.logger import logging


class ThresholdManager:

    def __init__(self, config: ThresholdManagerConfig):
        self.config = config

    def calculate_threshold(self, scores: np.ndarray):
        if self.config.threshold_method == "percentile":
            return float(
                np.percentile(
                    scores,
                    self.config.percentile
                )
            )

        raise ValueError(
            f"Unsupported threshold method: "
            f"{self.config.threshold_method}"
        )

    def initiate_threshold_management(self):

        try:
            score_dir = Path(self.config.input_score_dir)
            output_dir = Path(self.config.output_prediction_dir)

            train_dir = score_dir / "train"
            test_dir = score_dir / "test"

            output_test_dir = output_dir / "test"
            output_test_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            train_files = sorted(train_dir.glob("*.npy"))

            thresholds = {}

            for train_file in train_files:

                machine_name = train_file.stem
                test_file = test_dir / train_file.name

                if not test_file.exists():
                    raise FileNotFoundError(
                        f"Test score file not found for {machine_name}"
                    )

                train_scores = np.load(train_file)
                test_scores = np.load(test_file)

                threshold = self.calculate_threshold(
                    train_scores
                )

                test_predictions = (
                    test_scores > threshold
                ).astype(int)

                np.save(
                    output_test_dir / f"{machine_name}.npy",
                    test_predictions
                )

                thresholds[machine_name] = threshold

                logging.info(
                    f"Threshold for {machine_name}: "
                    f"{threshold:.6f} | "
                    f"Anomalous windows: "
                    f"{test_predictions.sum()}"
                )

            threshold_file = (
                Path(self.config.root_dir)
                / "thresholds.npy"
            )

            np.save(
                threshold_file,
                thresholds,
                allow_pickle=True
            )

            logging.info(
                "Threshold management completed successfully."
            )

        except Exception as e:
            raise CustomException(e)