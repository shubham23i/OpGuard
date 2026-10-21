
from opsguard_ai.configuration.manager import ConfigurationManager
from opsguard_ai.components.prediction import PredictionPipeline
from opsguard_ai.utils.logger import logging
from opsguard_ai.utils.exception import CustomException
import sys
class OpsGuardPredictionPipeline:

    def __init__(self):
        self.config_manager = ConfigurationManager()

        self.prediction_config = (
            self.config_manager.get_prediction_config()
        )

        self.predictor = PredictionPipeline(
            self.prediction_config
        )

    def initiate_prediction_pipeline(self, machine_names=None, split="test"):
        try:
            input_dir = self.prediction_config.input_data_dir

            if machine_names is None:
                machine_names = sorted(
                    path.stem
                    for path in (
                        input_dir / split
                    ).glob("*.txt")
                )

            if not machine_names:
                raise ValueError(
                    f"No machine data found for split: {split}"
                )

            results = []

            for machine_name in machine_names:
                result = self.predictor.predict_machine(
                    machine_name=machine_name,
                    split=split
                )
                results.append(result)

            summary = {
                "status": "SUCCESS",
                "split": split,
                "machines_processed": len(results),
                "total_anomalous_windows": sum(
                    result["anomaly_count"] for result in results
                ),
                "results": results
            }

            logging.info(
                "Prediction pipeline completed for %s machines",
                len(results)
            )

            return summary

        except Exception as e:
            raise CustomException(e,sys)
            
