
import json
import shutil
from datetime import datetime, timezone
import sys
from opsguard_ai.utils.logger import logging
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.common import create_directories


class ModelRegistry:
    def __init__(self, config):
        self.config = config

    def _next_version(self):
        versions = []

        if self.config.registry_dir.exists():
            for path in self.config.registry_dir.iterdir():
                if path.is_dir() and path.name.startswith("v"):
                    try:
                        versions.append(int(path.name[1:]))
                    except ValueError:
                        continue

        return max(versions, default=0) + 1

    def initiate_registration(self):
        try:
            if not self.config.model_dir.exists():
                raise FileNotFoundError(
                    f"Model directory not found: {self.config.model_dir}"
                )

            model_files = list(self.config.model_dir.glob("*.pt"))

            if not model_files:
                raise FileNotFoundError(
                    "No trained .pt model files found."
                )

            if not self.config.threshold_file.exists():
                raise FileNotFoundError(
                    f"Threshold file not found: {self.config.threshold_file}"
                )

            version_number = self._next_version()
            version_name = f"v{version_number}"
            version_dir = self.config.registry_dir / version_name

            create_directories([
                version_dir / "models",
                version_dir / "reports",
            ])

            copied_models = []

            for model_file in model_files:
                shutil.copy2(
                    model_file,
                    version_dir / "models" / model_file.name,
                )
                copied_models.append(model_file.name)

            shutil.copy2(
                self.config.threshold_file,
                version_dir / "thresholds.npy",
            )

            if self.config.evaluation_report_dir.exists():
                for report in self.config.evaluation_report_dir.glob("*.json"):
                    shutil.copy2(
                        report,
                        version_dir / "reports" / report.name,
                    )

            metadata = {
                "version": version_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model_type": "GRU Autoencoder",
                "model_count": len(copied_models),
                "models": copied_models,
                "threshold_file": "thresholds.npy",
                "evaluation_reports_included": (
                    len(list((version_dir / "reports").glob("*.json")))
                ),
            }

            version_metadata = version_dir / "metadata.json"

            with open(
                version_metadata, "w", encoding="utf-8"
            ) as file:
                json.dump(metadata, file, indent=4)

            registry = []

            if self.config.metadata_file.exists():
                with open(
                    self.config.metadata_file, "r", encoding="utf-8"
                ) as file:
                    registry = json.load(file)

            registry.append(metadata)

            with open(
                self.config.metadata_file, "w", encoding="utf-8"
            ) as file:
                json.dump(registry, file, indent=4)

            logging.info(
                f"Model artifacts registered successfully as {version_name}"
            )

            return metadata

        except Exception as e:
            raise CustomException(e,sys)
