from opsguard_ai.pipelines.training_pipeline import TrainingPipeline

pipeline = TrainingPipeline()
result = pipeline.initiate_training_pipeline()

print(result)