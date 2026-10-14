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
    AnomalyDetectionConfig,
    ThresholdManagerConfig,
    EventCorrelatorConfig,
    RootCauseConfig,
    UncertaintyConfig,
    DriftDetectionConfig,
    ModelEvaluationConfig,
    ModelRegistryConfig,
    PredictionConfig,
    IncidentManagerConfig,
    DecisionEngineConfig,
    AlertManagerConfig
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

    def get_threshold_manager_config(self) -> ThresholdManagerConfig:
        config = self.config.threshold_manager

        create_directories([
            config.root_dir,
            config.output_prediction_dir
        ])

        return ThresholdManagerConfig(
            root_dir=Path(config.root_dir),
            input_score_dir=Path(config.input_score_dir),
            output_prediction_dir=Path(config.output_prediction_dir),
            threshold_method=config.threshold_method,
            percentile=config.percentile
        )

    def get_event_correlator_config(self) -> EventCorrelatorConfig:
        config = self.config.event_correlator

        create_directories([
            config.root_dir,
            config.output_event_dir
        ])

        return EventCorrelatorConfig(
            root_dir=Path(config.root_dir),
            input_prediction_dir=Path(config.input_prediction_dir),
            output_event_dir=Path(config.output_event_dir),
            max_gap=config.max_gap,
            min_event_length=config.min_event_length
        )

    def get_root_cause_config(self) -> RootCauseConfig:
        config = self.config.root_cause

        create_directories([
            config.root_dir,
            config.output_dir
        ])

        return RootCauseConfig(
            root_dir=Path(config.root_dir),
            input_window_dir=Path(config.input_window_dir),
            input_score_dir=Path(config.input_score_dir),
            input_event_dir=Path(config.input_event_dir),
            model_dir=Path(config.model_dir),
            output_dir=Path(config.output_dir),
            top_k=config.top_k,
            batch_size=config.batch_size
        )

    def get_uncertainty_config(self) -> UncertaintyConfig:
        config = self.config.uncertainty

        create_directories([
            config.root_dir,
            config.output_dir
        ])

        return UncertaintyConfig(
            root_dir=Path(config.root_dir),
            input_window_dir=Path(config.input_window_dir),
            input_score_dir=Path(config.input_score_dir),
            output_dir=Path(config.output_dir),
            contamination=config.contamination,
            percentile=config.percentile,
            random_state=config.random_state
        )

    
    def get_drift_detection_config(self) -> DriftDetectionConfig:
        config = self.config.drift_detection

        create_directories([config.root_dir, config.output_dir])

        return DriftDetectionConfig(
            root_dir=Path(config.root_dir),
            input_data_dir=Path(config.input_data_dir),
            output_dir=Path(config.output_dir),
            reference_split=config.reference_split,
            current_split=config.current_split,
            method=config.method,
            psi_threshold=float(config.psi_threshold),
            bins=int(config.bins),
            epsilon=float(config.epsilon),
        )
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([
            config.root_dir,
            config.output_dir,
        ])

        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            input_label_dir=Path(config.input_label_dir),
            input_prediction_dir=Path(config.input_prediction_dir),
            output_dir=Path(config.output_dir),
        )

    
    def get_model_registry_config(self) -> ModelRegistryConfig:
        config = self.config.model_registry

        create_directories([
            config.root_dir,
            config.registry_dir,
        ])

        return ModelRegistryConfig(
            root_dir=Path(config.root_dir),
            model_dir=Path(config.model_dir),
            threshold_file=Path(config.threshold_file),
            evaluation_report_dir=Path(config.evaluation_report_dir),
            registry_dir=Path(config.registry_dir),
            metadata_file=Path(config.metadata_file),
        )
    
    def get_prediction_config(self) -> PredictionConfig:
        config = self.config.prediction

        create_directories([
            config.root_dir,
            config.output_dir,
        ])

        return PredictionConfig(
            root_dir=Path(config.root_dir),
            input_data_dir=Path(config.input_data_dir),
            scaler_dir=Path(config.scaler_dir),
            model_dir=Path(config.model_dir),
            threshold_file=Path(config.threshold_file),
            output_dir=Path(config.output_dir),
            window_size=int(config.window_size),
            hidden_size=int(config.hidden_size),
            latent_size=int(config.latent_size),
            num_layers=int(config.num_layers),
            dropout=float(config.dropout),
            batch_size=int(config.batch_size),
        )

    
    def get_incident_manager_config(self) -> IncidentManagerConfig:
        config = self.config.incident_manager

        create_directories([
            config.root_dir,
            config.output_dir,
        ])

        return IncidentManagerConfig(
            root_dir=Path(config.root_dir),
            input_event_dir=Path(config.input_event_dir),
            input_root_cause_dir=Path(config.input_root_cause_dir),
            input_uncertainty_dir=Path(config.input_uncertainty_dir),
            input_explainability_dir=Path(config.input_explainability_dir),
            output_dir=Path(config.output_dir),
            high_risk_threshold=float(config.high_risk_threshold),
            medium_risk_threshold=float(config.medium_risk_threshold),
        )

    def get_decision_engine_config(self) -> DecisionEngineConfig:
        config = self.config.decision_engine

        create_directories([config.root_dir, config.output_dir])

        return DecisionEngineConfig(
            root_dir=Path(config.root_dir),
            input_incident_dir=Path(config.input_incident_dir),
            output_dir=Path(config.output_dir),
            high_priority_threshold=config.high_priority_threshold,
            medium_priority_threshold=config.medium_priority_threshold,
            duration_weight=config.duration_weight,
            ood_weight=config.ood_weight,
        )
    def get_alert_manager_config(self) -> AlertManagerConfig:
        config = self.config.alert_manager

        create_directories([config.root_dir, config.output_dir])

        return AlertManagerConfig(
            root_dir=Path(config.root_dir),
            input_decisions_file=Path(config.input_decisions_file),
            output_dir=Path(config.output_dir),
            deduplicate=config.deduplicate,
        )