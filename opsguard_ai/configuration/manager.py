from pathlib import Path

from opsguard_ai.constants import CONFIG_FILE_PATH, SCHEMA_FILE_PATH
from opsguard_ai.entities.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataPreprocessingConfig,
    TemporalWindowingConfig,
    FeatureEngineeringConfig,
    LabelAlignmentConfig,
    BaselineDetectionConfig,
    AnomalyDetectionConfig
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
            data_dir=Path(config.data_dir),
            expected_machine_count=config.expected_machine_count,
            expected_feature_count=config.expected_feature_count
        )

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:

        config = self.config.data_preprocessing

        create_directories([
            config.root_dir,
            config.processed_data_dir,
            config.scaler_dir
        ])

        return DataPreprocessingConfig(
            root_dir=Path(config.root_dir),
            input_data_dir=Path(config.input_data_dir),
            processed_data_dir=Path(config.processed_data_dir),
            scaler_dir=Path(config.scaler_dir)
        )

    def get_temporal_windowing_config(self) -> TemporalWindowingConfig:

        config = self.config.temporal_windowing

        create_directories([
            config.root_dir,
            config.output_data_dir
        ])

        return TemporalWindowingConfig(
            root_dir=Path(config.root_dir),
            input_data_dir=Path(config.input_data_dir),
            output_data_dir=Path(config.output_data_dir),
            window_size=config.window_size,
            stride=config.stride
        )

    def get_label_alignment_config(self) -> LabelAlignmentConfig:

        config = self.config.label_alignment

        create_directories([
            config.root_dir,
            config.output_dir
        ])

        return LabelAlignmentConfig(
            root_dir=Path(config.root_dir),
            input_window_dir=Path(config.input_window_dir),
            input_label_dir=Path(config.input_label_dir),
            output_dir=Path(config.output_dir),
            window_size=config.window_size,
            anomaly_rule=config.anomaly_rule
        )

    def get_feature_engineering_config(self) -> FeatureEngineeringConfig:
        config = self.config.feature_engineering

        create_directories([
            config.root_dir,
            config.output_data_dir
        ])

        return FeatureEngineeringConfig(
            root_dir=Path(config.root_dir),
            input_data_dir=Path(config.input_data_dir),
            output_data_dir=Path(config.output_data_dir),
            window_size=config.window_size,
            rolling_windows=config.rolling_windows
        )

    def get_baseline_detection_config(self) -> BaselineDetectionConfig:
        config = self.config.baseline_detection

        create_directories([
            config.root_dir,
            config.output_data_dir,
            config.model_dir
        ])

        return BaselineDetectionConfig(
            root_dir=Path(config.root_dir),
            input_data_dir=Path(config.input_data_dir),
            output_data_dir=Path(config.output_data_dir),
            model_dir=Path(config.model_dir),
            contamination=config.contamination,
            random_state=config.random_state
        )

    def get_anomaly_detection_config(self) -> AnomalyDetectionConfig:
        config = self.config.anomaly_detection

        create_directories([
            config.root_dir,
            config.model_dir,
            config.score_dir
        ])

        return AnomalyDetectionConfig(
            root_dir=Path(config.root_dir),
            input_data_dir=Path(config.input_data_dir),
            model_dir=Path(config.model_dir),
            score_dir=Path(config.score_dir),
            hidden_size=config.hidden_size,
            latent_size=config.latent_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            epochs=config.epochs,
            patience=config.patience,
            random_state=config.random_state
        )