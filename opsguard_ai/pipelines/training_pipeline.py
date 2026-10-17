from opsguard_ai.configuration.manager import ConfigurationManager
from opsguard_ai.components.data_ingestion import DataIngestion
from opsguard_ai.components.data_validation import DataValidation
from opsguard_ai.components.data_preprocessing import DataPreprocessing
from opsguard_ai.components.temporal_windowing import TemporalWindowing
from opsguard_ai.components.feature_engineering import FeatureEngineering
from opsguard_ai.components.baseline_detector import BaselineDetector
from opsguard_ai.components.anomaly_detector import AnomalyDetector
from opsguard_ai.components.threshold_manager import ThresholdManager
from opsguard_ai.components.event_correlator import EventCorrelator
from opsguard_ai.components.root_cause import RootCauseAnalyzer
from opsguard_ai.components.uncertainty import UncertaintyAnalyzer
from opsguard_ai.components.explainability import Explainability
from opsguard_ai.components.drift_detector import DriftDetector
from opsguard_ai.components.model_evaluation import ModelEvaluation
from opsguard_ai.components.model_registry import ModelRegistry
from opsguard_ai.components.pipeline_monitoring import PipelineMonitor


class TrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        self.monitor = PipelineMonitor(
            self.config.get_pipeline_monitoring_config()
        )

    def run_stage(self, name, component, method_name):
        import time

        start = time.perf_counter()

        try:
            method = getattr(component, method_name)
            result = method()

            self.monitor.record_execution(
                component_name=name,
                status="SUCCESS",
                duration_seconds=time.perf_counter() - start,
            )

            return result

        except Exception as e:
            try:
                self.monitor.record_execution(
                    component_name=name,
                    status="FAILED",
                    duration_seconds=time.perf_counter() - start,
                    error=str(e),
                )
            except Exception:
                pass

            raise

    def initiate_training_pipeline(self):
        stages = [
            (
                "data_ingestion",
                DataIngestion(self.config.get_data_ingestion_config()),
                "initiate_data_ingestion",
            ),
            (
                "data_validation",
                DataValidation(self.config.get_data_validation_config()),
                "initiate_data_validation",
            ),
            (
                "data_preprocessing",
                DataPreprocessing(self.config.get_data_preprocessing_config()),
                "initiate_data_preprocessing",
            ),
            (
                "temporal_windowing",
                TemporalWindowing(self.config.get_temporal_windowing_config()),
                "initiate_windowing",
            ),
            (
                "feature_engineering",
                FeatureEngineering(self.config.get_feature_engineering_config()),
                "initiate_feature_engineering",
            ),
            (
                "baseline_detection",
                BaselineDetector(self.config.get_baseline_detection_config()),
                "initiate_baseline_detection",
            ),
            (
                "anomaly_detection",
                AnomalyDetector(self.config.get_anomaly_detection_config()),
                "initiate_anomaly_detection",
            ),
            (
                "threshold_manager",
                ThresholdManager(self.config.get_threshold_manager_config()),
                "initiate_threshold_management",
            ),
            (
                "event_correlator",
                EventCorrelator(self.config.get_event_correlator_config()),
                "initiate_event_correlation",
            ),
            (
                "root_cause",
                RootCauseAnalyzer(self.config.get_root_cause_config()),
                "initiate_root_cause_analysis",
            ),
            (
                "uncertainty",
                UncertaintyAnalyzer(self.config.get_uncertainty_config()),
                "initiate_uncertainty_analysis",
            ),
            (
                "explainability",
                Explainability(self.config.get_explainability_config()),
                "initiate_explainability",
            ),
            (
                "drift_detection",
                DriftDetector(self.config.get_drift_detection_config()),
                "initiate_drift_detection",
            ),
            (
                "model_evaluation",
                ModelEvaluation(self.config.get_model_evaluation_config()),
                "initiate_model_evaluation",
            ),
            (
                "model_registry",
                ModelRegistry(self.config.get_model_registry_config()),
                "initiate_model_registry",
            ),
        ]

        results = {}

        for name, component, method_name in stages:
            results[name] = self.run_stage(
                name,
                component,
                method_name,
            )

        return results