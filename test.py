from opsguard_ai.configuration.manager import ConfigurationManager
from opsguard_ai.components.anomaly_detector import AnomalyDetector


config = ConfigurationManager()

anomaly_config = config.get_anomaly_detection_config()

anomaly_detector = AnomalyDetector(
    config=anomaly_config
)

anomaly_detector.initiate_anomaly_detection()

print("GRU Autoencoder completed.")