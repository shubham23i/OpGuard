import os
import sys
from pathlib import Path
import logging

project_name = "OpsGuard-AI"

list_of_files =[
    # Configuration
    f"{project_name}/config/config.yaml",
    f"{project_name}/config/schema.yaml",

    # Main package
    f"{project_name}/opsguard_ai/__init__.py",

    # Components
    f"{project_name}/opsguard_ai/components/__init__.py",
    f"{project_name}/opsguard_ai/components/data_ingestion.py",
    f"{project_name}/opsguard_ai/components/data_validation.py",
    f"{project_name}/opsguard_ai/components/data_preprocessing.py",
    f"{project_name}/opsguard_ai/components/temporal_windowing.py",
    f"{project_name}/opsguard_ai/components/feature_engineering.py",
    f"{project_name}/opsguard_ai/components/baseline_detector.py",
    f"{project_name}/opsguard_ai/components/anomaly_detector.py",
    f"{project_name}/opsguard_ai/components/threshold_manager.py",
    f"{project_name}/opsguard_ai/components/event_correlator.py",
    f"{project_name}/opsguard_ai/components/root_cause.py",
    f"{project_name}/opsguard_ai/components/uncertainty.py",
    f"{project_name}/opsguard_ai/components/explainability.py",
    f"{project_name}/opsguard_ai/components/drift_detector.py",
    f"{project_name}/opsguard_ai/components/incident_manager.py",
    f"{project_name}/opsguard_ai/components/decision_engine.py",

    # Pipelines
    f"{project_name}/opsguard_ai/pipelines/__init__.py",
    f"{project_name}/opsguard_ai/pipelines/training_pipeline.py",
    f"{project_name}/opsguard_ai/pipelines/prediction_pipeline.py",

    # Configuration
    f"{project_name}/opsguard_ai/configuration/__init__.py",
    f"{project_name}/opsguard_ai/configuration/manager.py",

    # Entities
    f"{project_name}/opsguard_ai/entities/__init__.py",
    f"{project_name}/opsguard_ai/entities/config_entity.py",
    f"{project_name}/opsguard_ai/entities/artifact_entity.py",

    # Constants
    f"{project_name}/opsguard_ai/constants/__init__.py",

    # Utilities
    f"{project_name}/opsguard_ai/utils/__init__.py",
    f"{project_name}/opsguard_ai/utils/common.py",
    f"{project_name}/opsguard_ai/utils/logger.py",
    f"{project_name}/opsguard_ai/utils/exception.py",

    # Tests
    f"{project_name}/tests/__init__.py",
    f"{project_name}/tests/test_data_validation.py",
    f"{project_name}/tests/test_preprocessing.py",
    f"{project_name}/tests/test_features.py",
    f"{project_name}/tests/test_anomaly_detection.py",
    f"{project_name}/tests/test_event_correlation.py",
    f"{project_name}/tests/test_prediction.py",

    # Notebooks
    f"{project_name}/notebooks/01_data_exploration.ipynb",
    f"{project_name}/notebooks/02_time_series_analysis.ipynb",
    f"{project_name}/notebooks/03_baseline_models.ipynb",
    f"{project_name}/notebooks/04_anomaly_detection.ipynb",
    f"{project_name}/notebooks/05_threshold_analysis.ipynb",
    f"{project_name}/notebooks/06_event_correlation.ipynb",
    f"{project_name}/notebooks/07_model_evaluation.ipynb",

    # Application
    "app.py",
    "streamlit_app.py",
    "test_prediction.py",

    # Project files
    "requirements.txt",
    "setup.py",
    "Dockerfile",
    ".dockerignore",
    ".gitignore",
    "README.md",
    "LICENSE",
    ]

for file_name in list_of_files:
    file_path=Path(file_name)
    filedir, filename= os.path.split(file_path)

    if filedir!="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory {filedir}")

    if(not os.path.exists(file_path) or os.path.getsize(file_path)==0):
        with open(file_path, "w") as f:
            pass
            logging.info(f"Creating file {file_path}")
    else:
        logging.info(f"File {file_path} already exists")