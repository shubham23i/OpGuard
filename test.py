
from opsguard_ai.configuration.manager import ConfigurationManager
from opsguard_ai.components.incident_manager import IncidentManager


def main():
    config = ConfigurationManager().get_incident_manager_config()
    manager = IncidentManager(config)

    print(manager.initiate_incident_management())


if __name__ == "__main__":
    main()
