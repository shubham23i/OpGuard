from opsguard_ai.configuration.manager import ConfigurationManager
from opsguard_ai.components.alert_manager import AlertManager


if __name__ == "__main__":
    config = ConfigurationManager().get_alert_manager_config()
    manager = AlertManager(config)

    alerts = manager.initiate_alert_manager()

    print(f"Total alerts: {len(alerts)}")

    for alert in alerts[:5]:
        print(
            alert["alert_id"],
            alert["priority"],
            alert["machine"],
            alert["delivery_status"],
        )