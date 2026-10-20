
import time

from opsguard_ai.configuration.manager import ConfigurationManager
from opsguard_ai.components.data_ingestion import DataIngestion
from opsguard_ai.components.data_validation import DataValidation
from opsguard_ai.components.data_preprocessing import DataPreprocessing
from opsguard_ai.components.temporal_windowing import TemporalWindowing
from opsguard_ai.components.label_alignment import LabelAlignment
from opsguard_ai.components.feature_engineering import FeatureEngineering
from opsguard_ai.components.anomaly_detector import AnomalyDetector
from opsguard_ai.components.threshold_manager import ThresholdManager
from opsguard_ai.components.baseline_detector import BaselineDetector
from opsguard_ai.components.drift_detector import DriftDetector
from opsguard_ai.components.model_evaluation import ModelEvaluation
from opsguard_ai.components.model_registry import ModelRegistry
from opsguard_ai.components.pipeline_monitoring import PipelineMonitor
from opsguard_ai.components.event_correlator import EventCorrelator
from opsguard_ai.components.root_cause import RootCauseAnalyzer
from opsguard_ai.components.uncertainty import UncertaintyAnalyzer
from opsguard_ai.components.explainability import Explainability
from opsguard_ai.components.incident_manager import IncidentManager
from opsguard_ai.components.decision_engine import DecisionEngine
from opsguard_ai.components.alert_manager import AlertManager
from opsguard_ai.utils.logger import logging


class TrainingPipeline:

    def __init__(self):
        self.config_manager = ConfigurationManager()

        self.monitor = PipelineMonitor(
            self.config_manager.get_pipeline_monitoring_config()
        )

    def _run_stage(self, name, component, method_name):
        start_time = time.perf_counter()

        try:
            logging.info("Starting pipeline stage: %s", name)

            result = getattr(component, method_name)()

            self.monitor.record_execution(
                component_name=name,
                status="SUCCESS",
                duration_seconds=time.perf_counter() - start_time
            )

            logging.info("Completed pipeline stage: %s", name)
            return result

        except Exception as e:
            try:
                self.monitor.record_execution(
                    component_name=name,
                    status="FAILED",
                    duration_seconds=time.perf_counter() - start_time,
                    error=str(e)
                )
            except Exception as monitor_error:
                logging.error(
                    "Failed to record monitoring status for %s: %s",
                    name,
                    monitor_error
                )

            logging.exception("Pipeline stage failed: %s", name)
            raise

    def initiate_training_pipeline(self):
        logging.info("Starting OpsGuard AI training pipeline")

        completed_stages = []

        stages = [
            (
                "data_ingestion",
                DataIngestion(
                    self.config_manager.get_data_ingestion_config()
                ),
                "initiate_data_ingestion"
            ),
            (
                "data_validation",
                DataValidation(
                    self.config_manager.get_data_validation_config()
                ),
                "validate_dataset"
            ),
            (
                "data_preprocessing",
                DataPreprocessing(
                    self.config_manager.get_data_preprocessing_config()
                ),
                "initiate_data_preprocessing"
            ),
            (
                "temporal_windowing",
                TemporalWindowing(
                    self.config_manager.get_temporal_windowing_config()
                ),
                "initiate_temporal_windowing"
            ),
            (
                "label_alignment",
                LabelAlignment(
                    self.config_manager.get_label_alignment_config()
                ),
                "initiate_label_alignment"
            ),
            (
                "feature_engineering",
                FeatureEngineering(
                    self.config_manager.get_feature_engineering_config()
                ),
                "initiate_feature_engineering"
            ),
            (
                "baseline_detection",
                BaselineDetector(
                    self.config_manager.get_baseline_detection_config()
                ),
                "initiate_baseline_detection"
            ),
            (
                "anomaly_detection",
                AnomalyDetector(
                    self.config_manager.get_anomaly_detection_config()
                ),
                "initiate_anomaly_detection"
            ),
            (
                "threshold_management",
                ThresholdManager(
                    self.config_manager.get_threshold_manager_config()
                ),
                "initiate_threshold_management"
            ),
            (
                "drift_detection",
                DriftDetector(
                    self.config_manager.get_drift_detection_config()
                ),
                "initiate_drift_detection"
            ),
            (
                "model_evaluation",
                ModelEvaluation(
                    self.config_manager.get_model_evaluation_config()
                ),
                "initiate_model_evaluation"
            ),
            (
                "model_registry",
                ModelRegistry(
                    self.config_manager.get_model_registry_config()
                ),
                "initiate_registration"
            ),
            (
                "event_correlation",
                EventCorrelator(
                    self.config_manager.get_event_correlator_config()
                ),
                "initiate_event_correlation"
            ),
            (
                "root_cause_analysis",
                RootCauseAnalyzer(
                    self.config_manager.get_root_cause_config()
                ),
                "initiate_root_cause_analysis"
            ),
            (
                "uncertainty_analysis",
                UncertaintyAnalyzer(
                    self.config_manager.get_uncertainty_config()
                ),
                "initiate_uncertainty_analysis"
            ),
            (
                "explainability",
                Explainability(
                    self.config_manager.get_explainability_config()
                ),
                "initiate_explainability"
            ),
            (
                "incident_management",
                IncidentManager(
                    self.config_manager.get_incident_manager_config()
                ),
                "initiate_incident_management"
            ),
            (
                "decision_engine",
                DecisionEngine(
                    self.config_manager.get_decision_engine_config()
                ),
                "initiate_decision_engine"
            ),
            (
                "alert_management",
                AlertManager(
                    self.config_manager.get_alert_manager_config()
                ),
                "initiate_alert_manager"
            )
        ]

        for name, component, method_name in stages:
            result = self._run_stage(name, component, method_name)

            if name == "data_validation" and result is not True:
                raise ValueError(
                    "Dataset validation failed. Training pipeline stopped."
                )

            completed_stages.append(name)

        logging.info("OpsGuard AI training pipeline completed successfully")

        return {
            "status": "SUCCESS",
            "completed_stages": completed_stages
        }
