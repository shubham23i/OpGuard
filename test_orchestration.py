from opsguard_ai.pipelines.ops_orchestration_pipeline import (
    OpsOrchestrationPipeline,
)


def validate_environment():
    return {"status": "healthy"}


def check_artifacts():
    return {"status": "available"}


if __name__ == "__main__":
    pipeline = OpsOrchestrationPipeline()

    results = pipeline.run([
        ("environment_check", validate_environment),
        ("artifact_check", check_artifacts),
    ])

    print(results)