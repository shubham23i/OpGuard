from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_data_dir: Path
    ingested_data_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    status_file: Path
    data_dir: Path
    expected_machine_count: int
    expected_feature_count: int


@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    input_data_dir: Path
    processed_data_dir: Path
    scaler_dir: Path


@dataclass(frozen=True)
class TemporalWindowingConfig:
    root_dir: Path
    input_data_dir: Path
    output_data_dir: Path
    window_size: int
    stride: int

@dataclass(frozen=True)
class LabelAlignmentConfig:
    root_dir: Path
    input_window_dir: Path
    input_label_dir: Path
    output_dir: Path
    window_size: int
    anomaly_rule: str


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    root_dir: Path
    input_data_dir: Path
    output_data_dir: Path
    window_size: int
    rolling_windows: list

@dataclass(frozen=True)
class BaselineDetectionConfig:
    root_dir: Path
    input_data_dir: Path
    output_data_dir: Path
    model_dir: Path
    contamination: float
    random_state: int

@dataclass(frozen=True)
class AnomalyDetectionConfig:
    root_dir: Path
    input_data_dir: Path
    model_dir: Path
    score_dir: Path
    hidden_size: int
    latent_size: int
    num_layers: int
    dropout: float
    learning_rate: float
    batch_size: int
    epochs: int
    patience: int
    random_state: int


@dataclass(frozen=True)
class ThresholdManagerConfig:
    root_dir: Path
    input_score_dir: Path
    output_prediction_dir: Path
    threshold_method: str
    percentile: float   