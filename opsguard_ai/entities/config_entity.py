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
    processed_data_dir: Path
    scaler_path: Path


@dataclass(frozen=True)
class TemporalWindowingConfig:
    root_dir: Path
    window_size: int
    stride: int


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    root_dir: Path
    processed_data_dir: Path