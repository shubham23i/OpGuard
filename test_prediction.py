
from opsguard_ai.pipelines.prediction_pipeline import (
    OpsGuardPredictionPipeline
)

pipeline = OpsGuardPredictionPipeline()

result = pipeline.initiate_prediction_pipeline(split="test")

print({
    "status": result["status"],
    "split": result["split"],
    "machines_processed": result["machines_processed"],
    "total_anomalous_windows": result["total_anomalous_windows"]
})
