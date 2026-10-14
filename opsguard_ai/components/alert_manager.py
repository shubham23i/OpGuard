import json
from datetime import datetime, timezone
import sys
from opsguard_ai.entities.config_entity import AlertManagerConfig
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.logger import logging


class AlertManager:
    def __init__(self, config: AlertManagerConfig):
        self.config = config

    def _load_decisions(self):
        path = self.config.input_decisions_file

        if not path.exists():
            raise FileNotFoundError(f"Decisions file not found: {path}")

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("decisions.json must contain a list.")

        return data

    def _load_existing_alerts(self, path):
        if not path.exists():
            return []

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("Existing alerts.json must contain a list.")

        return data

    def initiate_alert_manager(self):
        try:
            logging.info("Starting Alert Manager")

            decisions = self._load_decisions()
            alerts_file = self.config.output_dir / "alerts.json"
            existing_alerts = self._load_existing_alerts(alerts_file)

            existing_ids = {
                str(alert.get("incident_id"))
                for alert in existing_alerts
            }

            alerts = list(existing_alerts)
            created_count = 0
            suppressed_count = 0

            for decision in decisions:
                incident_id = decision.get("incident_id")

                if not incident_id:
                    logging.warning("Skipping decision without incident_id")
                    continue

                incident_id = str(incident_id)

                if self.config.deduplicate and incident_id in existing_ids:
                    suppressed_count += 1
                    continue

                priority = str(
                    decision.get("priority", "MEDIUM")
                ).upper()

                if priority not in {"HIGH", "MEDIUM", "LOW"}:
                    priority = "MEDIUM"

                channel = (
                    "PAGERDUTY_OR_ON_CALL"
                    if priority == "HIGH"
                    else "SLACK_OR_EMAIL"
                    if priority == "MEDIUM"
                    else "DASHBOARD"
                )

                alert = {
                    "alert_id": f"ALERT-{incident_id}",
                    "incident_id": incident_id,
                    "machine": decision.get("machine", "UNKNOWN"),
                    "priority": priority,
                    "priority_score": decision.get("priority_score"),
                    "recommended_action": decision.get(
                        "recommended_action",
                        "Review the incident manually.",
                    ),
                    "delivery_channel": channel,
                    "delivery_status": "PENDING_INTEGRATION",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "status": "OPEN",
                }

                alerts.append(alert)
                existing_ids.add(incident_id)
                created_count += 1

            with open(alerts_file, "w", encoding="utf-8") as file:
                json.dump(alerts, file, indent=4)

            summary = {
                "total_alerts": len(alerts),
                "new_alerts_created": created_count,
                "duplicates_suppressed": suppressed_count,
                "open_alerts": sum(
                    alert.get("status") == "OPEN" for alert in alerts
                ),
                "delivery_note": (
                    "Alerts are recorded locally; external delivery "
                    "has not been configured."
                ),
            }

            summary_file = self.config.output_dir / "summary.json"

            with open(summary_file, "w", encoding="utf-8") as file:
                json.dump(summary, file, indent=4)

            logging.info(
                "Alert Manager completed: %s new alerts, %s duplicates suppressed",
                created_count,
                suppressed_count,
            )

            return alerts

        except Exception as e:
            logging.exception("Alert Manager failed")
            raise CustomException(e,sys)