
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)

from opsguard_ai.entities.config_entity import ThresholdManagerConfig
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.logger import logging


class ThresholdManager:

    CALIBRATION_FRACTION = 0.30

    def __init__(self, config: ThresholdManagerConfig):
        self.config = config

    def calculate_threshold(self, scores: np.ndarray):
        scores = np.asarray(scores, dtype=float).ravel()
        scores = scores[np.isfinite(scores)]

        if len(scores) == 0:
            raise ValueError("No finite scores available for threshold calculation.")

        if self.config.threshold_method == "percentile":
            return float(
                np.percentile(scores, self.config.percentile)
            )

        raise ValueError(
            f"Unsupported threshold method: "
            f"{self.config.threshold_method}"
        )

    def calibrate_threshold(
        self,
        train_scores: np.ndarray,
        calibration_scores: np.ndarray,
        calibration_labels: np.ndarray,
    ):
        calibration_scores = np.asarray(
            calibration_scores, dtype=float
        ).ravel()
        calibration_labels = np.asarray(
            calibration_labels, dtype=int
        ).ravel()

        if len(calibration_scores) != len(calibration_labels):
            raise ValueError("Calibration scores and labels are misaligned.")

        if not np.isfinite(calibration_scores).all():
            raise ValueError("Calibration scores contain non-finite values.")

        if not np.isin(calibration_labels, [0, 1]).all():
            raise ValueError("Calibration labels must contain only 0 and 1.")

        if np.unique(calibration_labels).size < 2:
            logging.warning(
                "Calibration segment contains only one class. "
                "Falling back to the training-score percentile."
            )
            return self.calculate_threshold(train_scores), "training_percentile"

        precision, recall, thresholds = precision_recall_curve(
            calibration_labels,
            calibration_scores,
        )

        if len(thresholds) == 0:
            return self.calculate_threshold(train_scores), "training_percentile"

        denominator = precision[:-1] + recall[:-1]
        f1_values = np.divide(
            2 * precision[:-1] * recall[:-1],
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )

        best_index = int(np.argmax(f1_values))
        threshold = float(thresholds[best_index])

        return threshold, "calibration_f1"

    @staticmethod
    def calculate_metrics(labels, predictions, scores):
        labels = np.asarray(labels, dtype=int).ravel()
        predictions = np.asarray(predictions, dtype=int).ravel()
        scores = np.asarray(scores, dtype=float).ravel()

        if not (
            len(labels) == len(predictions) == len(scores)
        ):
            raise ValueError("Evaluation labels, predictions and scores are misaligned.")

        tn, fp, fn, tp = confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).ravel()

        pr_auc = (
            float(average_precision_score(labels, scores))
            if np.unique(labels).size == 2
            else None
        )

        return {
            "windows_evaluated": int(len(labels)),
            "anomaly_rate": float(labels.mean()) if len(labels) else 0.0,
            "precision": float(precision_score(
                labels, predictions, zero_division=0
            )),
            "recall": float(recall_score(
                labels, predictions, zero_division=0
            )),
            "f1_score": float(f1_score(
                labels, predictions, zero_division=0
            )),
            "pr_auc": pr_auc,
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }

    @staticmethod
    def save_json(path: Path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, allow_nan=False)

    def initiate_threshold_management(self):

        try:
            root_dir = Path(self.config.root_dir)
            score_dir = Path(self.config.input_score_dir)
            output_dir = Path(self.config.output_prediction_dir)

            train_dir = score_dir / "train"
            test_dir = score_dir / "test"

            # Existing full-length prediction directory.
            output_test_dir = output_dir / "test"
            output_test_dir.mkdir(parents=True, exist_ok=True)

            # Aligned labels for the test windows.
            label_dir = root_dir.parent / "label_alignment" / "labels"

            # New calibration and evaluation artifacts.
            validation_root = root_dir / "validation"
            evaluation_root = root_dir / "evaluation"

            validation_dirs = {
                "labels": validation_root / "labels",
                "scores": validation_root / "scores",
                "predictions": validation_root / "predictions",
            }

            evaluation_dirs = {
                "labels": evaluation_root / "labels",
                "scores": evaluation_root / "scores",
                "predictions": evaluation_root / "predictions",
            }

            for directory in (
                *validation_dirs.values(),
                *evaluation_dirs.values(),
            ):
                directory.mkdir(parents=True, exist_ok=True)

            train_files = sorted(train_dir.glob("*.npy"))

            if not train_files:
                raise FileNotFoundError(
                    f"No training score files found in {train_dir}"
                )

            thresholds = {}
            threshold_sources = {}
            evaluation_reports = {}
            split_manifest = {}

            for train_file in train_files:
                machine_name = train_file.stem

                test_file = test_dir / train_file.name
                label_file = label_dir / train_file.name

                if not test_file.exists():
                    raise FileNotFoundError(
                        f"Test score file not found for {machine_name}"
                    )

                if not label_file.exists():
                    raise FileNotFoundError(
                        f"Aligned label file not found for {machine_name}: "
                        f"{label_file}"
                    )

                train_scores = np.asarray(
                    np.load(train_file), dtype=float
                ).ravel()
                test_scores = np.asarray(
                    np.load(test_file), dtype=float
                ).ravel()
                labels = np.asarray(
                    np.load(label_file), dtype=int
                ).ravel()

                if len(test_scores) != len(labels):
                    raise ValueError(
                        f"{machine_name}: test scores ({len(test_scores)}) "
                        f"and labels ({len(labels)}) have different lengths."
                    )

                if len(test_scores) < 2:
                    raise ValueError(
                        f"{machine_name}: insufficient test windows."
                    )

                if not np.isfinite(train_scores).all():
                    raise ValueError(
                        f"{machine_name}: training scores contain non-finite values."
                    )

                if not np.isfinite(test_scores).all():
                    raise ValueError(
                        f"{machine_name}: test scores contain non-finite values."
                    )

                if not np.isin(labels, [0, 1]).all():
                    raise ValueError(
                        f"{machine_name}: labels must contain only 0 and 1."
                    )

                # Chronological split: first 30% for calibration.
                split_index = int(
                    len(test_scores) * self.CALIBRATION_FRACTION
                )
                split_index = max(1, min(split_index, len(test_scores) - 1))

                calibration_scores = test_scores[:split_index]
                calibration_labels = labels[:split_index]

                evaluation_scores = test_scores[split_index:]
                evaluation_labels = labels[split_index:]

                threshold, threshold_source = self.calibrate_threshold(
                    train_scores,
                    calibration_scores,
                    calibration_labels,
                )

                # Preserve the original full-length prediction output.
                full_predictions = (
                    test_scores > threshold
                ).astype(np.int8)

                np.save(
                    output_test_dir / train_file.name,
                    full_predictions,
                )

                # Save calibration artifacts for auditing.
                np.save(
                    validation_dirs["scores"] / train_file.name,
                    calibration_scores,
                )
                np.save(
                    validation_dirs["labels"] / train_file.name,
                    calibration_labels,
                )
                np.save(
                    validation_dirs["predictions"] / train_file.name,
                    full_predictions[:split_index],
                )

                # Evaluation predictions are created using the fixed
                # calibration threshold; evaluation labels do not tune it.
                evaluation_predictions = (
                    evaluation_scores > threshold
                ).astype(np.int8)

                np.save(
                    evaluation_dirs["scores"] / train_file.name,
                    evaluation_scores,
                )
                np.save(
                    evaluation_dirs["labels"] / train_file.name,
                    evaluation_labels,
                )
                np.save(
                    evaluation_dirs["predictions"] / train_file.name,
                    evaluation_predictions,
                )

                thresholds[machine_name] = threshold
                threshold_sources[machine_name] = threshold_source

                evaluation_reports[machine_name] = self.calculate_metrics(
                    evaluation_labels,
                    evaluation_predictions,
                    evaluation_scores,
                )

                split_manifest[machine_name] = {
                    "total_windows": int(len(test_scores)),
                    "calibration_start": 0,
                    "calibration_end_exclusive": int(split_index),
                    "evaluation_start": int(split_index),
                    "evaluation_end_exclusive": int(len(test_scores)),
                    "calibration_windows": int(split_index),
                    "evaluation_windows": int(len(evaluation_scores)),
                    "calibration_anomaly_rate": float(
                        calibration_labels.mean()
                    ),
                    "evaluation_anomaly_rate": float(
                        evaluation_labels.mean()
                    ),
                    "threshold": float(threshold),
                    "threshold_source": threshold_source,
                }

                logging.info(
                    f"{machine_name} | "
                    f"threshold={threshold:.6f} | "
                    f"source={threshold_source} | "
                    f"calibration={len(calibration_scores)} | "
                    f"evaluation={len(evaluation_scores)} | "
                    f"evaluation F1="
                    f"{evaluation_reports[machine_name]['f1_score']:.4f}"
                )

            # Preserve the existing threshold dictionary format.
            threshold_file = root_dir / "thresholds.npy"
            np.save(
                threshold_file,
                thresholds,
                allow_pickle=True,
            )

            self.save_json(
                root_dir / "threshold_sources.json",
                threshold_sources,
            )
            self.save_json(
                root_dir / "split_manifest.json",
                split_manifest,
            )

            # Only the held-out final 70% contributes to these metrics.
            valid_reports = list(evaluation_reports.values())

            metric_names = [
                "precision", "recall", "f1_score", "pr_auc"
            ]
            summary = {
                "machines_evaluated": len(valid_reports),
                "calibration_fraction": self.CALIBRATION_FRACTION,
                "evaluation_fraction": 1.0 - self.CALIBRATION_FRACTION,
                "threshold_selection": "calibration_segment_max_f1",
                "reported_metrics_use": "evaluation_segment_only",
            }

            for metric in metric_names:
                values = [
                    report[metric]
                    for report in valid_reports
                    if report[metric] is not None
                ]
                summary[f"mean_{metric}"] = (
                    float(np.mean(values)) if values else None
                )

            self.save_json(
                evaluation_root / "reports" / "per_machine.json",
                evaluation_reports,
            )
            self.save_json(
                evaluation_root / "reports" / "summary.json",
                summary,
            )

            logging.info(
                "Chronological threshold calibration completed. "
                "Full-length predictions preserved; evaluation metrics "
                "computed on the held-out final segment."
            )

        except Exception as e:
            raise CustomException(e,sys)
