from opsguard_ai.configuration.manager import ConfigurationManager
from opsguard_ai.components.data_preprocessing import DataPreprocessing


config = ConfigurationManager()

data_preprocessing_config = (
    config.get_data_preprocessing_config()
)

data_preprocessing = DataPreprocessing(
    config=data_preprocessing_config
)

data_preprocessing.initiate_data_preprocessing()

print("Data preprocessing completed.")