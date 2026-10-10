
from opsguard_ai.configuration.manager import ConfigurationManager
from opsguard_ai.components.prediction import PredictionPipeline


def main():
    config = ConfigurationManager().get_prediction_config()
    pipeline = PredictionPipeline(config)

    result = pipeline.predict_machine(
        machine_name="machine-1-1",
        split="test",
    )

    print({
        "machine": result["machine"],
        "window_count": result["window_count"],
        "anomaly_count": result["anomaly_count"],
        "anomaly_fraction": result["anomaly_fraction"],
    })


if __name__ == "__main__":
    main()
