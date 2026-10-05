from opsguard_ai.configuration.manager import ConfigurationManager
from opsguard_ai.components.root_cause import RootCauseAnalyzer


config = ConfigurationManager()

root_cause_config = config.get_root_cause_config()

analyzer = RootCauseAnalyzer(
    config=root_cause_config
)

analyzer.initiate_root_cause_analysis()

print("Root-cause candidate analysis completed.")