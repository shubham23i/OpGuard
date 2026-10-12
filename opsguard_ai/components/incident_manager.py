
import json
from pathlib import Path
import sys
from opsguard_ai.utils.logger import logging
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.common import create_directories


class IncidentManager:
    def __init__(self, config):
        self.config = config

    def _read_json(self, path):
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _find_machine_file(self, directory, machine):
        if not directory.exists():
            return None

        candidates = list(directory.glob(f"{machine}*.json"))
        return candidates[0] if candidates else None

    def _get_severity(self, risk_score):
        if risk_score >= self.config.high_risk_threshold:
            return "HIGH"
        if risk_score >= self.config.medium_risk_threshold:
            return "MEDIUM"
        return "LOW"

    def initiate_incident_management(self):
        try:
            if not self.config.input_event_dir.exists():
                raise FileNotFoundError(
                    f"Event directory not found: "
                    f"{self.config.input_event_dir}"
                )

            event_files = sorted(
                self.config.input_event_dir.glob("*.json")
            )

            if not event_files:
                raise FileNotFoundError(
                    "No event JSON files found."
                )

            create_directories([self.config.output_dir])

            incidents = []
            incident_counter = 0

            for event_file in event_files:
                event_data = self._read_json(event_file)

                if not event_data:
                    continue

                machine = event_data.get(
                    "machine", event_file.stem
                )

                events = event_data.get("events", [])

                if isinstance(event_data, list):
                    events = event_data

                if isinstance(event_data, dict) and not events:
                    if "start" in event_data or "start_window" in event_data:
                        events = [event_data]

                root_cause_file = self._find_machine_file(
                    self.config.input_root_cause_dir, machine
                )
                explainability_file = self._find_machine_file(
                    self.config.input_explainability_dir, machine
                )

                root_cause_data = self._read_json(
                    root_cause_file
                ) if root_cause_file else None

                explainability_data = self._read_json(
                    explainability_file
                ) if explainability_file else None

                uncertainty_path = (
                    self.config.input_uncertainty_dir
                    / f"{machine}.npy"
                )

                ood_detected = None

                if uncertainty_path.exists():
                    import numpy as np

                    uncertainty = np.load(uncertainty_path)

                    if uncertainty.ndim == 2 and uncertainty.shape[1] >= 3:
                        ood_detected = bool(
                            (uncertainty[:, 2] == 1).any()
                        )

                for event in events:
                    incident_counter += 1

                    start = event.get(
                        "start_window", event.get("start")
                    )
                    end = event.get(
                        "end_window", event.get("end")
                    )

                    anomaly_count = event.get(
                        "length",
                        event.get("duration", None)
                    )

                    if anomaly_count is None and start is not None and end is not None:
                        anomaly_count = max(0, end - start + 1)

                    root_candidates = (
                        root_cause_data.get(
                            "events",
                            root_cause_data.get("candidates", [])
                        )
                        if isinstance(root_cause_data, dict)
                        else []
                    )

                    incident = {
                        "incident_id": f"INC-{incident_counter:05d}",
                        "machine": machine,
                        "start_window": start,
                        "end_window": end,
                        "anomaly_window_count": anomaly_count,
                        "severity": (
                            "HIGH" if ood_detected is True else "MEDIUM"
                        ),
                        "ood_detected": ood_detected,
                        "root_cause_candidates": root_candidates,
                        "explainability_report": (
                            str(explainability_file)
                            if explainability_file else None
                        ),
                        "status": "OPEN",
                        "recommended_action": (
                            "Investigate sensor behavior and validate "
                            "the top-ranked metric contributions."
                        ),
                    }

                    incidents.append(incident)

            output_path = self.config.output_dir / "incidents.json"

            with open(
                output_path, "w", encoding="utf-8"
            ) as file:
                json.dump(incidents, file, indent=4)

            summary = {
                "incidents_created": len(incidents),
                "output_file": str(output_path),
                "status": "completed",
            }

            with open(
                self.config.output_dir / "summary.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(summary, file, indent=4)

            logging.info(
                f"Incident management completed: "
                f"{len(incidents)} incidents created."
            )

            return summary

        except Exception as e:
            raise CustomException(e,sys)
