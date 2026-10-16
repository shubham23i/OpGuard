import time
import sys
from opsguard_ai.configuration.manager import ConfigurationManager
from opsguard_ai.components.pipeline_monitoring import PipelineMonitor
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.logger import logging


class OpsOrchestrationPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.monitor = PipelineMonitor(
            self.config_manager.get_pipeline_monitoring_config()
        )

    def run_component(self, component_name, component_callable):
        start_time = time.perf_counter()

        try:
            logging.info("Starting component: %s", component_name)
            result = component_callable()
            duration = time.perf_counter() - start_time

            self.monitor.record_execution(
                component_name=component_name,
                status="SUCCESS",
                duration_seconds=duration,
            )

            logging.info(
                "Completed component: %s in %.2f seconds",
                component_name,
                duration,
            )
            return result

        except Exception as e:
            duration = time.perf_counter() - start_time

            try:
                self.monitor.record_execution(
                    component_name=component_name,
                    status="FAILED",
                    duration_seconds=duration,
                    error=str(e),
                )
            except Exception:
                logging.exception("Could not record component failure")

            logging.exception("Component failed: %s", component_name)
            raise CustomException(e,sys)

    def run(self, components):
        results = {}

        for component_name, component_callable in components:
            results[component_name] = self.run_component(
                component_name,
                component_callable,
            )

        return results