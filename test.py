from opsguard_ai.configuration.manager import ConfigurationManager
from opsguard_ai.components.pipeline_monitoring import PipelineMonitor


if __name__ == "__main__":
    config = ConfigurationManager().get_pipeline_monitoring_config()
    monitor = PipelineMonitor(config)

    monitor.record_execution(
        component_name="anomaly_detector",
        status="SUCCESS",
        duration_seconds=12.45,
    )

    monitor.record_execution(
        component_name="drift_detector",
        status="FAILED",
        duration_seconds=2.31,
        error="Test failure for monitoring validation",
    )

    print("Pipeline monitoring test completed.")
    print(f"History: {monitor.history_file}")
    print(f"Summary: {monitor.summary_file}")