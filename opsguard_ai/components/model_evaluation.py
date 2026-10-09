
import json

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)
import sys
from opsguard_ai.utils.logger import logging
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.common import create_directories


class ModelEvaluation:
    def __init__(self, config):
        self.config = config

    def initiate_model_evaluation(self):
        try:
            label_files = {
                path.stem: path
                for path in self.config.input_label_dir.glob("*.npy")
            }

            prediction_files = {
                path.stem: path
                for path in self.config.input_prediction_dir.glob("*.npy")
            }

            score_dir = (
                self.config.input_prediction_dir.parent.parent
                / "scores"
                / "test"
            )

            # Standard OpsGuard score location:
            # artifacts/anomaly_detection/scores/test
            if not score_dir.exists():
                score_dir = (
                    self.config.input_prediction_dir.parent.parent.parent
                    / "anomaly_detection"
                    / "scores"
                    / "test"
                )

            score_files = {
                path.stem: path
                for path in score_dir.glob("*.npy")
            } if score_dir.exists() else {}

            machines = sorted(
                set(label_files)
                & set(prediction_files)
            )

            if not machines:
                raise FileNotFoundError(
                    "No matching label and prediction .npy files found."
                )

            create_directories([self.config.output_dir])
            reports = []

            for machine in machines:
                y_true = np.asarray(
                    np.load(label_files[machine])
                ).reshape(-1)

                prediction_data = np.asarray(
                    np.load(prediction_files[machine])
                )

                if prediction_data.ndim == 2:
                    # Threshold-manager files contain:
                    # [predicted_label, anomaly_score]
                    y_pred = prediction_data[:, 0].astype(int)
                else:
                    y_pred = prediction_data.reshape(-1).astype(int)

                if len(y_true) != len(y_pred):
                    raise ValueError(
                        f"Length mismatch for {machine}: "
                        f"labels={len(y_true)}, "
                        f"predictions={len(y_pred)}"
                    )

                if not np.isin(y_true, [0, 1]).all():
                    raise ValueError(
                        f"Labels for {machine} are not binary."
                    )

                if not np.isin(y_pred, [0, 1]).all():
                    raise ValueError(
                        f"Predictions for {machine} are not binary."
                    )

                precision = precision_score(
                    y_true, y_pred, zero_division=0
                )
                recall = recall_score(
                    y_true, y_pred, zero_division=0
                )
                f1 = f1_score(
                    y_true, y_pred, zero_division=0
                )

                tn, fp, fn, tp = confusion_matrix(
                    y_true, y_pred, labels=[0, 1]
                ).ravel()

                pr_auc = None

                if machine in score_files and len(np.unique(y_true)) > 1:
                    scores = np.asarray(
                        np.load(score_files[machine])
                    ).reshape(-1)

                    if len(scores) == len(y_true):
                        pr_auc = float(
                            average_precision_score(y_true, scores)
                        )

                report = {
                    "machine": machine,
                    "samples": int(len(y_true)),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "pr_auc": pr_auc,
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "true_negatives": int(tn),
                    "false_negatives": int(fn),
                }

                reports.append(report)

                with open(
                    self.config.output_dir / f"{machine}.json",
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(report, file, indent=4)

                logging.info(
                    f"Evaluated {machine}: F1={f1:.4f}"
                )

            summary = {
                "machines_evaluated": len(reports),
                "mean_precision": float(
                    np.mean([r["precision"] for r in reports])
                ),
                "mean_recall": float(
                    np.mean([r["recall"] for r in reports])
                ),
                "mean_f1_score": float(
                    np.mean([r["f1_score"] for r in reports])
                ),
                "mean_pr_auc": (
                    float(np.mean([
                        r["pr_auc"] for r in reports
                        if r["pr_auc"] is not None
                    ]))
                    if any(r["pr_auc"] is not None for r in reports)
                    else None
                ),
            }

            with open(
                self.config.output_dir / "summary.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(summary, file, indent=4)

            logging.info("Model evaluation completed.")
            return summary

        except Exception as e:
            raise CustomException(e,sys)