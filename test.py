
from opsguard_ai.configuration.manager import ConfigurationManager
from opsguard_ai.components.explainability import Explainability


if __name__ == "__main__":
    config = ConfigurationManager().get_explainability_config()
    component = Explainability(config)

    summary = component.initiate_explainability()

    print("Explainability completed.")
    print(summary)
    print(f"Reports saved to: {config.output_dir}")
