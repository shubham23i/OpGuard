from opsguard_ai.configuration.manager import ConfigurationManager
from opsguard_ai.components.decision_engine import DecisionEngine


if __name__ == "__main__":
    config = ConfigurationManager().get_decision_engine_config()
    engine = DecisionEngine(config)

    decisions = engine.initiate_decision_engine()

    print(f"Generated {len(decisions)} decisions")

    for decision in decisions[:5]:
        print(
            decision["machine"],
            decision["priority"],
            decision["priority_score"],
            decision["recommended_action"],
        )