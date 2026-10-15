import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from opsguard_ai.entities.config_entity import PipelineMonitoringConfig
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.logger import logging


class PipelineMonitor:
    def __init__(self, config: PipelineMonitoringConfig):
        self.config = config
        self.history_file = config.output_dir / "execution_history.json"
        self.summary_file = config.output_dir / "summary.json"

    def _load_history(self):
        if not self.history_file.exists():
            return []

        with open(self.history_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("Execution history must be a JSON list.")

        return data

    def record_execution(
        self,
        component_name: str,
        status: str,
        duration_seconds: float,
        error=None,
    ):
        try:
            if not component_name.strip():
                raise ValueError("component_name cannot be empty.")

            status = status.upper()
            if status not in {"SUCCESS", "FAILED", "SKIPPED"}:
                raise ValueError(
                    "status must be SUCCESS, FAILED, or SKIPPED."
                )

            if duration_seconds < 0:
                raise ValueError("duration_seconds cannot be negative.")

            record = {
                "component": component_name,
                "status": status,
                "duration_seconds": round(float(duration_seconds), 4),
                "error": str(error) if error is not None else None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            history = self._load_history()
            history.append(record)

            with open(self.history_file, "w", encoding="utf-8") as file:
                json.dump(history, file, indent=4)

            counts = {
                "SUCCESS": sum(
                    item.get("status") == "SUCCESS" for item in history
                ),
                "FAILED": sum(
                    item.get("status") == "FAILED" for item in history
                ),
                "SKIPPED": sum(
                    item.get("status") == "SKIPPED" for item in history
                ),
            }

            summary = {
                "total_executions": len(history),
                "successful_executions": counts["SUCCESS"],
                "failed_executions": counts["FAILED"],
                "skipped_executions": counts["SKIPPED"],
                "success_rate_percent": round(
                    counts["SUCCESS"] / len(history) * 100, 2
                ),
                "total_duration_seconds": round(
                    sum(
                        float(item.get("duration_seconds", 0))
                        for item in history
                    ),
                    4,
                ),
                "last_execution": record,
            }

            with open(self.summary_file, "w", encoding="utf-8") as file:
                json.dump(summary, file, indent=4)

            logging.info(
                "Component %s recorded with status %s",
                component_name,
                status,
            )

            return record

        except Exception as e:
            logging.exception("Failed to record pipeline execution")
            raise CustomException(e,sys)