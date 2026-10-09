from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
import sys
from opsguard_ai.entities.config_entity import UncertaintyConfig
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.logger import logging


class UncertaintyAnalyzer:

    def __init__(self, config: UncertaintyConfig):
        self.config = config

    @staticmethod
    def extract_window_features(windows):
        means = np.mean(windows, axis=1)
        stds = np.std(windows, axis=1)

        return np.concatenate([means, stds], axis=1)

    def initiate_uncertainty_analysis(self):
        try:
            input_dir = Path(self.config.input_window_dir)
            score_dir = Path(self.config.input_score_dir)
            output_dir = Path(self.config.output_dir)

            train_window_dir = input_dir / "train"
            test_window_dir = input_dir / "test"
            train_score_dir = score_dir / "train"
            test_score_dir = score_dir / "test"

            output_dir.mkdir(parents=True, exist_ok=True)

            train_files = sorted(train_window_dir.glob("*.npy"))

            if not train_files:
                raise FileNotFoundError(
                    f"No training windows found in {train_window_dir}"
                )

            for train_file in train_files:
                machine_name = train_file.stem

                test_file = test_window_dir / train_file.name
                train_scores_file = train_score_dir / train_file.name
                test_scores_file = test_score_dir / train_file.name

                for path in [
                    test_file,
                    train_scores_file,
                    test_scores_file
                ]:
                    if not path.exists():
                        raise FileNotFoundError(
                            f"Required artifact missing: {path}"
                        )

                train_windows = np.load(train_file).astype(np.float32)
                test_windows = np.load(test_file).astype(np.float32)

                train_scores = np.load(train_scores_file)
                test_scores = np.load(test_scores_file)

                if len(train_windows) != len(train_scores):
                    raise ValueError(
                        f"Training window/score mismatch: {machine_name}"
                    )

                if len(test_windows) != len(test_scores):
                    raise ValueError(
                        f"Test window/score mismatch: {machine_name}"
                    )

                train_features = self.extract_window_features(
                    train_windows
                )
                test_features = self.extract_window_features(
                    test_windows
                )

                model = IsolationForest(
                    contamination=self.config.contamination,
                    random_state=self.config.random_state,
                    n_jobs=-1
                )

                model.fit(train_features)

                train_ood_scores = -model.score_samples(train_features)
                test_ood_scores = -model.score_samples(test_features)

                threshold = float(
                    np.percentile(
                        train_ood_scores,
                        self.config.percentile
                    )
                )

                ood_flags = (
                    test_ood_scores > threshold
                ).astype(int)

                output = np.column_stack([
                    test_scores,
                    test_ood_scores,
                    ood_flags
                ])

                np.save(
                    output_dir / f"{machine_name}.npy",
                    output
                )

                logging.info(
                    f"OOD analysis completed for {machine_name} | "
                    f"OOD windows: {int(ood_flags.sum())} | "
                    f"Threshold: {threshold:.6f}"
                )

            logging.info(
                "Uncertainty and OOD analysis completed successfully."
            )

        except Exception as e:
            raise CustomException(e,sys)