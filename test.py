from opsguard_ai.configuration.manager import ConfigurationManager
from opsguard_ai.components.data_validation import DataValidation


config = ConfigurationManager()

data_validation_config = config.get_data_validation_config()

data_validation = DataValidation(
    config=data_validation_config
)

status = data_validation.validate_dataset()

print("Validation status:", status)