from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

            output_train_dir = (
                Path(self.config.processed_data_dir) / "train"
            )

            output_test_dir = (
                Path(self.config.processed_data_dir) / "test"
            )

            output_train_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            output_test_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            scaler_dir = Path(self.config.scaler_dir)
            scaler_dir.mkdir(
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
                )

                test_data = pd.read_csv(
                    test_file,
                    header=None
                )

                scaler = StandardScaler()

                train_scaled = scaler.fit_transform(
                    train_data
                )

                test_scaled = scaler.transform(
                    test_data
                )

                train_output = output_train_dir / train_file.name
                test_output = output_test_dir / test_file.name

                pd.DataFrame(train_scaled).to_csv(
                    train_output,
                    header=False,
                    index=False
                )

                pd.DataFrame(test_scaled).to_csv(
                    test_output,
                    header=False,
                    index=False
                )

                scaler_path = scaler_dir / f"{machine_name}.pkl"

                joblib.dump(
                    scaler,
                    scaler_path
                )

                logging.info(
                    f"Preprocessed machine: {machine_name}"
                )

            logging.info(
                "Data preprocessing completed successfully."
            )

        except Exception as e:
            raise CustomException(e)