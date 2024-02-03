import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from src.utils.dirs import DIRS
from src.experiments.mlflow_experiments import MLflowExperiment
from src.utils.logger import configure_logger


logger = configure_logger(name=os.path.basename(__file__)[:-3], log_level="INFO")


class BaselineModel:
    def __init__(self):
        with open(DIRS["config_file_path"], 'r') as file:
            self.config = yaml.safe_load(file)

    def prepare_data(self):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(DIRS["processed_data_file_path"])

        # Separate features (X) and target variable (y)
        X = df.drop(['is_canceled'], axis=1)
        y = df['is_canceled']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=self.config['train_test_split']['test_size'],
                                                            random_state=self.config['train_test_split']['random_state']
                                                            )

        return X_train, X_test, y_train, y_test

    def __call__(self):
        # Get the test data
        _, X_test, _, y_test = self.prepare_data()

        majority_class = y_test.mode()[0]

        # Create a predictions column with the majority class
        y_pred = pd.Series([majority_class] * len(X_test), name='baseline_predictions')

        # Reset the index of y_test to match predictions
        y_test = y_test.reset_index(drop=True)

        # Evaluate the baseline model
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        logger.info(f"Baseline Model Accuracy on Test Data: {accuracy:.2%}")

        metrics = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
        }

        mlflow_experiment = MLflowExperiment()
        mlflow_experiment('Baseline', X_test, None, None, metrics)


if __name__ == '__main__':
    baseline_model = BaselineModel()
    baseline_model()
    logger.info("Baseline Model execution completed successfully.")
