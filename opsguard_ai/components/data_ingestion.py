import shutil
from pathlib import Path

from opsguard_ai.entities.config_entity import DataIngestionConfig
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.logger import logging


class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):

        try:
            source_dir = Path(self.config.source_data_dir)
            destination_dir = Path(self.config.ingested_data_dir)

            if not source_dir.exists():
                raise FileNotFoundError(
                    f"SMD dataset not found at: {source_dir}"
                )

            if destination_dir.exists():
                shutil.rmtree(destination_dir)

            shutil.copytree(source_dir, destination_dir)

            logging.info(
                f"SMD dataset copied from {source_dir} "
                f"to {destination_dir}"
            )

            return destination_dir

        except Exception as e:
            raise CustomException(e)