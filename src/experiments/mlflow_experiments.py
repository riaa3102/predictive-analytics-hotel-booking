import os
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import yaml
from src.utils.dirs import DIRS
from src.utils.logger import configure_logger


logger = configure_logger(name=os.path.basename(__file__)[:-3], log_level="INFO")


class MLflowExperiment:
    def __init__(self):
        with open(DIRS["config_file_path"], 'r') as file:
            self.config = yaml.safe_load(file)

        try:
            # Set tracking server uri for logging
            mlflow.set_tracking_uri(uri=self.config['MLflow']['uri'])
            logger.info(f"Current tracking uri: {mlflow.get_tracking_uri()}")

            # Create a new MLflow Experiment
            mlflow.set_experiment(self.config['MLflow']['experiment_name'])

            # mlflow.sklearn.autolog()

            logger.info("MLflowExperiment initialized successfully.")

        except Exception as e:
            logger.error(f"Error initializing MLflowExperiment: {e}")

    def __call__(self, run_name, X_train, params, model, metrics):
        try:
            with mlflow.start_run(run_name=f"{run_name}"):
                if params:
                    mlflow.log_params(params)

                mlflow.log_metrics(metrics)

                mlflow.set_tag("Training Info", f"Basic {run_name} model for hotel data")

                sample_input = X_train.iloc[[0]]

                if model:
                    # Infer the model signature
                    signature = infer_signature(sample_input, model.predict(sample_input))

                    # Log the model with the signature
                    mlflow.sklearn.log_model(sk_model=model,
                                             artifact_path=self.config['MLflow']['artifact_path'],
                                             signature=signature,
                                             input_example=sample_input,
                                             registered_model_name=self.config['MLflow']['registered_model_name'],
                                             )

                run = mlflow.active_run()
                logger.info(f"Active run_id: {run.info.run_id}")

        except Exception as e:
            logger.error(f"Error logging MLflow experiment: {e}")


# if __name__ == '__main__':
#     mlflow_experiment = MLflowExperiment()
