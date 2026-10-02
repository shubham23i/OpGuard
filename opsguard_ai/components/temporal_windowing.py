from pathlib import Path

import numpy as np
import pandas as pd

from opsguard_ai.entities.config_entity import TemporalWindowingConfig
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.logger import logging


class TemporalWindowing:

    def __init__(self, config: TemporalWindowingConfig):
        self.config = config

    def create_windows(self, data: np.ndarray):

        window_size = self.config.window_size
        stride = self.config.stride

        windows = []

        for start in range(
            0,
            len(data) - window_size + 1,
            stride
        ):
            end = start + window_size
            windows.append(data[start:end])

        return np.asarray(windows)

    def initiate_temporal_windowing(self):

        try:
            input_dir = Path(self.config.input_data_dir)

            train_dir = input_dir / "train"
            test_dir = input_dir / "test"

            output_dir = Path(self.config.output_data_dir)

            output_train_dir = output_dir / "train"
            output_test_dir = output_dir / "test"

            output_train_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            output_test_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            train_files = sorted(train_dir.glob("*.txt"))

            for train_file in train_files:

                machine_name = train_file.stem

                test_file = test_dir / train_file.name

                if not test_file.exists():
                    raise FileNotFoundError(
                        f"Test file not found for {machine_name}"
                    )

                train_data = pd.read_csv(
                    train_file,
                    header=None
                ).values

                test_data = pd.read_csv(
                    test_file,
                    header=None
                ).values

                train_windows = self.create_windows(
                    train_data
                )

                test_windows = self.create_windows(
                    test_data
                )

                np.save(
                    output_train_dir / f"{machine_name}.npy",
                    train_windows
                )

                np.save(
                    output_test_dir / f"{machine_name}.npy",
                    test_windows
                )

                logging.info(
                    f"Created windows for {machine_name} | "
                    f"Train: {train_windows.shape} | "
                    f"Test: {test_windows.shape}"
                )

            logging.info(
                "Temporal windowing completed successfully."
            )

        except Exception as e:
            raise CustomException(e)