from opsguard_ai.configuration.manager import ConfigurationManager
from opsguard_ai.components.event_correlator import EventCorrelator


config = ConfigurationManager()

event_config = config.get_event_correlator_config()

event_correlator = EventCorrelator(
    config=event_config
)

event_correlator.initiate_event_correlation()

print("Event correlation completed.")