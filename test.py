
from opsguard_ai.configuration.manager import ConfigurationManager
from opsguard_ai.components.drift_detector import DriftDetector


def main():
    config = ConfigurationManager().get_drift_detection_config()
    detector = DriftDetector(config)

    summary = detector.initiate_drift_detection()

    print("Drift Detection Completed")
    print(summary)


if __name__ == "__main__":
    main()