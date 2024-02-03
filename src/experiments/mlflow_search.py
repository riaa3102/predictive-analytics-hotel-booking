import os
import mlflow
import yaml
import shutil
from src.utils.dirs import DIRS
from src.utils.logger import configure_logger


logger = configure_logger(name=os.path.basename(__file__)[:-3], log_level="INFO")


# Set the working directory
os.chdir(DIRS["mlexperiments_dir"])

with open(DIRS["config_file_path"], 'r') as file:
    config = yaml.safe_load(file)

# Set your experiment name
experiment_name = config['MLflow']['experiment_name']

# Load the MLflow experiment from the specified directory
experiment = mlflow.get_experiment_by_name(experiment_name)

# Get the path to the experiment directory
experiment_directory = experiment.artifact_location

# Query runs with the highest accuracy
best_run = mlflow.search_runs(experiment_ids=experiment.experiment_id,
                              order_by=["metrics.accuracy desc"]
                              ).iloc[0]

# Retrieve the run ID and get the model path
best_run_id = best_run.run_id
best_model_path = os.path.join(experiment_directory, best_run_id, "artifacts", config['MLflow']['artifact_path'])[8:]
best_model_path = os.path.normpath(best_model_path)

model_path = os.path.join(DIRS["models_dir"], config['MLflow']['artifact_path'])

# Copy the model file to DIRS["models_dir"]
shutil.copytree(best_model_path, model_path, dirs_exist_ok=True)

logger.info("Best model search completed successfully.")

