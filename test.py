from opsguard_ai.configuration.manager import ConfigurationManager
from opsguard_ai.components.threshold_manager import ThresholdManager


config = ConfigurationManager()

threshold_config = config.get_threshold_manager_config()

threshold_manager = ThresholdManager(
    config=threshold_config
)

threshold_manager.initiate_threshold_management()

print("Threshold management completed.")