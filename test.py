
from opsguard_ai.configuration.manager import ConfigurationManager
from opsguard_ai.components.model_evaluation import ModelEvaluation


def main():
    config = ConfigurationManager().get_model_evaluation_config()
    evaluator = ModelEvaluation(config)
    print(evaluator.initiate_model_evaluation())


if __name__ == "__main__":
    main()