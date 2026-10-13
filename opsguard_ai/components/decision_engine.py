import json
from pathlib import Path
import sys
from opsguard_ai.entities.config_entity import DecisionEngineConfig
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.logger import logging


class DecisionEngine:
    def __init__(self, config: DecisionEngineConfig):
        self.config = config

    def _load_incidents(self):
        incident_file = self.config.input_incident_dir / "incidents.json"

        if not incident_file.exists():
            raise FileNotFoundError(
                f"Incident file not found: {incident_file}"
            )

        with open(incident_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            incidents = data.get("incidents", [])
            if isinstance(incidents, list):
                return incidents

        raise ValueError("Unsupported incidents.json format.")

    def _calculate_priority_score(self, incident):
        severity = str(incident.get("severity", "MEDIUM")).upper()

        severity_scores = {
            "CRITICAL": 0.90,
            "HIGH": 0.75,
            "MEDIUM": 0.50,
            "LOW": 0.25,
        }

        score = severity_scores.get(severity, 0.50)

        try:
            start = int(incident.get("start_window", 0))
            end = int(incident.get("end_window", start))
            duration = max(0, end - start + 1)
        except (TypeError, ValueError):
            duration = 1

        duration_factor = min(duration / 100, 1.0)
        score += self.config.duration_weight * duration_factor

        ood_detected = incident.get("ood_detected", False)
        if isinstance(ood_detected, str):
            ood_detected = ood_detected.strip().lower() in {
                "true", "1", "yes"
            }

        if ood_detected:
            score += self.config.ood_weight

        return round(min(score, 1.0), 4)

    def _recommend_action(self, priority):
        actions = {
            "CRITICAL": (
                "Escalate immediately and investigate the affected machine."
            ),
            "HIGH": (
                "Investigate promptly and arrange a targeted inspection."
            ),
            "MEDIUM": (
                "Review anomaly evidence and schedule a maintenance check."
            ),
            "LOW": (
                "Continue monitoring and review during routine maintenance."
            ),
        }
        return actions[priority]

    def _get_priority(self, score):
        if score >= self.config.high_priority_threshold:
            return "HIGH"

        if score >= self.config.medium_priority_threshold:
            return "MEDIUM"

        return "LOW"

    def initiate_decision_engine(self):
        try:
            logging.info("Starting Decision Engine")

            incidents = self._load_incidents()
            decisions = []

            for incident in incidents:
                score = self._calculate_priority_score(incident)
                priority = self._get_priority(score)

                decision = {
                    "incident_id": incident.get("incident_id", "UNKNOWN"),
                    "machine": incident.get("machine", "UNKNOWN"),
                    "severity": incident.get("severity", "MEDIUM"),
                    "priority": priority,
                    "priority_score": score,
                    "ood_detected": incident.get("ood_detected", False),
                    "recommended_action": self._recommend_action(priority),
                    "reasoning": [
                        f"Incident severity: {incident.get('severity', 'MEDIUM')}",
                        f"Calculated priority score: {score}",
                        f"OOD signal detected: {incident.get('ood_detected', False)}",
                    ],
                    "status": "PENDING_REVIEW",
                }

                decisions.append(decision)

            decisions.sort(
                key=lambda item: item["priority_score"],
                reverse=True,
            )

            output_file = self.config.output_dir / "decisions.json"
            summary_file = self.config.output_dir / "summary.json"

            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(decisions, file, indent=4)

            summary = {
                "total_decisions": len(decisions),
                "high_priority": sum(
                    item["priority"] == "HIGH" for item in decisions
                ),
                "medium_priority": sum(
                    item["priority"] == "MEDIUM" for item in decisions
                ),
                "low_priority": sum(
                    item["priority"] == "LOW" for item in decisions
                ),
                "decisions_file": str(output_file),
            }

            with open(summary_file, "w", encoding="utf-8") as file:
                json.dump(summary, file, indent=4)

            logging.info("Decision Engine completed successfully")
            logging.info("Saved decisions to %s", output_file)

            return decisions

        except Exception as e:
            logging.exception("Decision Engine failed")
            raise CustomException(e,sys)