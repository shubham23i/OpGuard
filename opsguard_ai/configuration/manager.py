from pathlib import Path

from opsguard_ai.constants import CONFIG_FILE_PATH, SCHEMA_FILE_PATH
from opsguard_ai.entities.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataPreprocessingConfig,
    TemporalWindowingConfig,
    FeatureEngineeringConfig
)
from opsguard_ai.utils.common import read_yaml, create_directories


class ConfigurationManager:

    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:

        config = self.config.data_ingestion

        create_directories([config.root_dir])
        create_directories([config.ingested_data_dir])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_data_dir=Path(config.source_data_dir),
            ingested_data_dir=Path(config.ingested_data_dir)
        )

    def get_data_validation_config(self) -> DataValidationConfig:

        config = self.config.data_validation

        create_directories([config.root_dir])

        return DataValidationConfig(
            root_dir=Path(config.root_dir),
            status_file=Path(config.status_file),
            required_files=config.required_files
        )

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:

        config = self.config.data_preprocessing

        create_directories([config.root_dir])
        create_directories([config.processed_data_dir])

        return DataPreprocessingConfig(
            root_dir=Path(config.root_dir),
            processed_data_dir=Path(config.processed_data_dir),
            scaler_path=Path(config.scaler_path)
        )

    def get_temporal_windowing_config(self) -> TemporalWindowingConfig:

        config = self.config.temporal_windowing

        create_directories([config.root_dir])

        return TemporalWindowingConfig(
            root_dir=Path(config.root_dir),
            window_size=config.window_size,
            stride=config.stride
        )

    def get_feature_engineering_config(self) -> FeatureEngineeringConfig:

        config = self.config.feature_engineering

        create_directories([config.root_dir])
        create_directories([config.processed_data_dir])

        return FeatureEngineeringConfig(
            root_dir=Path(config.root_dir),
            processed_data_dir=Path(config.processed_data_dir)
        )