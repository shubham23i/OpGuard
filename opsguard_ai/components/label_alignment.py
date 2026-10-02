from pathlib import Path

import numpy as np
import pandas as pd

from opsguard_ai.entities.config_entity import LabelAlignmentConfig
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.logger import logging


class LabelAlignment:

    def __init__(self, config: LabelAlignmentConfig):
        self.config = config

    def create_window_labels(self, labels: np.ndarray):

        window_size = self.config.window_size

        window_labels = []

        for start in range(
            0,
            len(labels) - window_size + 1
        ):
            end = start + window_size

            window = labels[start:end]

            if self.config.anomaly_rule == "any":
                label = int(np.any(window == 1))

            elif self.config.anomaly_rule == "all":
                label = int(np.all(window == 1))

            else:
                raise ValueError(
                    f"Unsupported anomaly rule: "
                    f"{self.config.anomaly_rule}"
                )

            window_labels.append(label)

        return np.asarray(window_labels)

    def initiate_label_alignment(self):

        try:
            window_dir = Path(self.config.input_window_dir)
            label_dir = Path(self.config.input_label_dir)
            output_dir = Path(self.config.output_dir)

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            test_window_dir = window_dir / "test"

            window_files = sorted(
                test_window_dir.glob("*.npy")
            )

            for window_file in window_files:

                machine_name = window_file.stem

                label_file = label_dir / f"{machine_name}.txt"

                if not label_file.exists():
                    raise FileNotFoundError(
                        f"Label file not found for "
                        f"{machine_name}"
                    )

                windows = np.load(window_file)

                labels = pd.read_csv(
                    label_file,
                    header=None
                ).iloc[:, 0].to_numpy()

                expected_windows = len(windows)

                expected_labels = (
                    len(labels) - self.config.window_size + 1
                )

                if expected_windows != expected_labels:
                    raise ValueError(
                        f"Window/label mismatch for "
                        f"{machine_name}: "
                        f"{expected_windows} windows vs "
                        f"{expected_labels} labels"
                    )

                window_labels = self.create_window_labels(
                    labels
                )

                output_file = (
                    output_dir / f"{machine_name}.npy"
                )

                np.save(
                    output_file,
                    window_labels
                )

                logging.info(
                    f"Labels aligned for {machine_name} | "
                    f"Windows: {len(windows)} | "
                    f"Anomalous windows: "
                    f"{window_labels.sum()}"
                )

            logging.info(
                "Label alignment completed successfully."
            )

        except Exception as e:
            raise CustomException(e)