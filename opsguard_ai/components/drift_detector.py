
import json
import numpy as np
import sys
from opsguard_ai.utils.logger import logging
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.common import create_directories


class DriftDetector:
    def __init__(self, config):
        self.config = config

    
    def _load_data(self, path):
        data = np.loadtxt(
            path,
            delimiter=",",
            dtype=np.float64,
        )

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.ndim != 2:
            raise ValueError(
                f"Expected 2D data in {path}, got {data.shape}"
            )

        if not np.isfinite(data).all():
            raise ValueError(
                f"Non-finite values found in {path}"
            )

        return data

    def _calculate_psi(self, reference, current):
        reference = np.asarray(reference, dtype=np.float64)
        current = np.asarray(current, dtype=np.float64)

        reference = reference[np.isfinite(reference)]
        current = current[np.isfinite(current)]

        if len(reference) == 0 or len(current) == 0:
            return None

        quantiles = np.linspace(
            0, 1, self.config.bins + 1
        )
        edges = np.unique(
            np.quantile(reference, quantiles)
        )

        if len(edges) < 2:
            value = reference[0]
            edges = np.array([value - 1e-9, value + 1e-9])

        edges[0] = -np.inf
        edges[-1] = np.inf

        reference_counts, _ = np.histogram(reference, bins=edges)
        current_counts, _ = np.histogram(current, bins=edges)

        epsilon = self.config.epsilon

        reference_pct = np.clip(
            reference_counts / len(reference), epsilon, None
        )
        current_pct = np.clip(
            current_counts / len(current), epsilon, None
        )

        return float(
            np.sum(
                (current_pct - reference_pct)
                * np.log(current_pct / reference_pct)
            )
        )

    def initiate_drift_detection(self):
        try:
            reference_dir = (
                self.config.input_data_dir
                / self.config.reference_split
            )
            current_dir = (
                self.config.input_data_dir
                / self.config.current_split
            )

            if not reference_dir.exists():
                raise FileNotFoundError(
                    f"Reference directory not found: {reference_dir}"
                )

            if not current_dir.exists():
                raise FileNotFoundError(
                    f"Current directory not found: {current_dir}"
                )

            reference_files = {
                path.stem: path
                for path in reference_dir.glob("*.txt")
            }
            current_files = {
                path.stem: path
                for path in current_dir.glob("*.txt")
            }

            common_machines = sorted(
                set(reference_files) & set(current_files)
            )

            if not common_machines:
                raise FileNotFoundError(
                    "No matching machine .txt files found."
                )

            create_directories([self.config.output_dir])

            summary = {
                "machines_processed": 0,
                "missing_current_files": sorted(
                    set(reference_files) - set(current_files)
                ),
                "missing_reference_files": sorted(
                    set(current_files) - set(reference_files)
                ),
                "output_dir": str(self.config.output_dir),
            }

            for machine in common_machines:
                logging.info(f"Calculating drift for {machine}")

                reference = self._load_data(reference_files[machine])
                current = self._load_data(current_files[machine])

                if reference.shape[1] != current.shape[1]:
                    raise ValueError(
                        f"Feature mismatch for {machine}: "
                        f"train={reference.shape[1]}, "
                        f"test={current.shape[1]}"
                    )

                features = []

                for index in range(reference.shape[1]):
                    psi = self._calculate_psi(
                        reference[:, index],
                        current[:, index],
                    )

                    drift = (
                        psi > self.config.psi_threshold
                        if psi is not None
                        else None
                    )

                    features.append({
                        "feature_index": index,
                        "psi": psi,
                        "drift_detected": drift,
                        "status": (
                            "insufficient_data"
                            if psi is None
                            else "drift" if drift else "stable"
                        ),
                    })

                valid = [
                    item for item in features
                    if item["psi"] is not None
                ]
                drifted = sum(
                    item["drift_detected"] is True
                    for item in features
                )

                report = {
                    "machine": machine,
                    "method": self.config.method,
                    "reference_split": self.config.reference_split,
                    "current_split": self.config.current_split,
                    "reference_samples": len(reference),
                    "current_samples": len(current),
                    "feature_count": reference.shape[1],
                    "psi_threshold": self.config.psi_threshold,
                    "drifted_feature_count": drifted,
                    "drift_fraction": (
                        drifted / len(valid) if valid else None
                    ),
                    "machine_drift_detected": drifted > 0,
                    "features": features,
                }

                output_path = (
                    self.config.output_dir / f"{machine}.json"
                )
                with open(output_path, "w", encoding="utf-8") as file:
                    json.dump(
                        report, file, indent=4, allow_nan=False
                    )

                summary["machines_processed"] += 1

            with open(
                self.config.output_dir / "summary.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(summary, file, indent=4, allow_nan=False)

            logging.info("Drift detection completed successfully.")
            return summary

        except Exception as e:
            raise CustomException(e,sys)