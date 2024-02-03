import os
from pathlib import Path
import pandas as pd
import mlflow
import yaml
from src.data.preprocess import DataPreprocessor
from src.data.transform import DataTransformer
from src.utils.dirs import DIRS
from src.utils.logger import configure_logger


logger = configure_logger(name=os.path.basename(__file__)[:-3], log_level="INFO")


class MLflowModel:
    def __init__(self):
        with open(DIRS["config_file_path"], 'r') as file:
            self.config = yaml.safe_load(file)

        self.preprocessor = DataPreprocessor()
        self.transformer = DataTransformer(load_flag=True)
        self.model_path = os.path.join(DIRS["models_dir"], self.config['MLflow']['artifact_path'])
        self.model = None

    def load_model(self):
        try:
            # Load the MLflow model
            self.model = mlflow.pyfunc.load_model(self.model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.info(f"Error loading the model: {e}")

    def predict(self, input_data):
        if self.model is None:
            logger.info("Model not loaded. Loading now...")
            self.load_model()

        # Perform necessary preprocessing
        input_data = self.preprocessor(input_data, save_csv=False)
        logger.info("Preprocessing done successfully.")

        # Perform necessary transformations
        _, input_data = self.transformer(X_test=input_data)
        logger.info("Transformations done successfully.")

        # Make predictions using the loaded model
        prediction = self.model.predict(input_data)
        logger.info("Prediction done successfully.")
        return prediction


if __name__ == "__main__":
    model = MLflowModel()
    input_data_path = Path(__file__).resolve().parent.parent.parent / "app_input_example" / "app_input_example.csv"
    input_data = pd.read_csv(input_data_path)
    # logger.info(input_data.head())
    prediction = model.predict(input_data)
    logger.info(f"prediction = {prediction}")

