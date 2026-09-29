from opsguard_ai.configuration.manager import ConfigurationManager
from opsguard_ai.components.data_ingestion import DataIngestion


config = ConfigurationManager()
data_ingestion_config = config.get_data_ingestion_config()

data_ingestion = DataIngestion(config=data_ingestion_config)
data_ingestion.initiate_data_ingestion()