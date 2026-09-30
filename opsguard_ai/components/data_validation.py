from pathlib import Path
import sys
import numpy as np
import pandas as pd

from opsguard_ai.entities.config_entity import DataValidationConfig
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.logger import logging


class DataValidation:

    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_dataset(self):

        try:
            data_dir = Path(self.config.data_dir)

            train_dir = data_dir / "train"
            test_dir = data_dir / "test"
            label_dir = data_dir / "test_label"

            status = True

            train_files = sorted(train_dir.glob("*.txt"))
            test_files = sorted(test_dir.glob("*.txt"))
            label_files = sorted(label_dir.glob("*.txt"))

            print("\n========== SMD DATA VALIDATION ==========")

            print(f"Dataset path     : {data_dir}")
            print(f"Train files      : {len(train_files)}")
            print(f"Test files       : {len(test_files)}")
            print(f"Label files      : {len(label_files)}")

            # Machine count
            if len(train_files) != self.config.expected_machine_count:
                print(
                    f"[FAIL] Expected 28 train files, "
                    f"found {len(train_files)}"
                )
                status = False
            else:
                print("[PASS] Train machine count")

            if len(test_files) != self.config.expected_machine_count:
                print(
                    f"[FAIL] Expected 28 test files, "
                    f"found {len(test_files)}"
                )
                status = False
            else:
                print("[PASS] Test machine count")

            if len(label_files) != self.config.expected_machine_count:
                print(
                    f"[FAIL] Expected 28 label files, "
                    f"found {len(label_files)}"
                )
                status = False
            else:
                print("[PASS] Label machine count")

            # Machine names
            train_names = {file.stem for file in train_files}
            test_names = {file.stem for file in test_files}
            label_names = {file.stem for file in label_files}

            if train_names != test_names:
                print("[FAIL] Train/test machine names do not match")
                print("Train only:", train_names - test_names)
                print("Test only :", test_names - train_names)
                status = False
            else:
                print("[PASS] Train/test machine names")

            if test_names != label_names:
                print("[FAIL] Test/label machine names do not match")
                print("Test only  :", test_names - label_names)
                print("Label only :", label_names - test_names)
                status = False
            else:
                print("[PASS] Test/label machine names")

            # Inspect files
            print("\n========== FILE INSPECTION ==========")

            for train_file in train_files:

                train_data = pd.read_csv(
                    train_file,
                    header=None
                )

                if train_data.shape[1] != self.config.expected_feature_count:

                    print(
                        f"[FAIL] {train_file.name}: "
                        f"{train_data.shape[1]} columns"
                    )

                    status = False

                if train_data.isnull().values.any():

                    print(
                        f"[FAIL] {train_file.name}: "
                        f"missing values"
                    )

                    status = False

                if not np.isfinite(train_data.to_numpy()).all():

                    print(
                        f"[FAIL] {train_file.name}: "
                        f"non-finite values"
                    )

                    status = False

            print("\n========== LABEL INSPECTION ==========")

            for label_file in label_files:

                labels = pd.read_csv(
                    label_file,
                    header=None
                )

                if labels.isnull().values.any():

                    print(
                        f"[FAIL] {label_file.name}: "
                        f"missing labels"
                    )

                    status = False

                unique_labels = set(labels.iloc[:, 0].unique())

                if not unique_labels.issubset({0, 1}):

                    print(
                        f"[FAIL] {label_file.name}: "
                        f"labels = {unique_labels}"
                    )

                    status = False

            self._write_status(status)

            print("\n=======================================")
            print(f"FINAL VALIDATION STATUS: {status}")
            print("=======================================\n")

            return status

        except Exception as e:
            raise CustomException(e)

    def _write_status(self, status: bool):

        status_file = Path(self.config.status_file)

        status_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(status_file, "w") as file:
            file.write(str(status))