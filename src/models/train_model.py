import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from argparse import ArgumentParser
from src.utils.dirs import DIRS
from src.data.transform import DataTransformer
from src.visualization.visualize import DataVisualizer
from src.experiments.mlflow_experiments import MLflowExperiment
from src.utils.logger import configure_logger


logger = configure_logger(name=os.path.basename(__file__)[:-3], log_level="INFO")


class TrainModel:
    def __init__(self):
        with open(DIRS["config_file_path"], 'r') as file:
            self.config = yaml.safe_load(file)

        self.transformer = DataTransformer()
        self.visualizer = DataVisualizer()
        self.mlflow_experiment = MLflowExperiment()

    def prepare_data(self):

        # Read the CSV file into a DataFrame
        df = pd.read_csv(DIRS["processed_data_file_path"])

        # Separate features (X) and target variable (y)
        X = df.drop([self.config['transform']['target_column']], axis=1)
        y = df[self.config['transform']['target_column']]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=self.config['train_test_split']['test_size'],
                                                            random_state=self.config['train_test_split']['random_state']
                                                            )

        X_train, X_test = self.transformer(X_train, X_test, y_train)

        return X_train, X_test, y_train, y_test

    def train_model(self, model_name, params, model, X_train, y_train, X_test, y_test):

        logger.info(f"{model_name} Model training started.")

        # Train the model using grid search
        grid_search = GridSearchCV(model, self.config[model_name]['param_grid'], **self.config['GridSearchCV'])
        grid_search.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = grid_search.best_estimator_.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        conf_mtx = confusion_matrix(y_test, y_pred)
        clf_report = classification_report(y_test, y_pred)

        logger.info(f"{model_name} Model Accuracy: {accuracy:.2%}")
        logger.info(f"{model_name} Model ROC AUC: {roc_auc:.2%}")
        logger.info(f"{model_name} Model Confusion Matrix : \n{conf_mtx}")
        logger.info(f"{model_name} Model Classification Report : \n{clf_report}")

        metrics = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
        }

        self.visualizer.plot_roc_curve(model_name, y_test, y_pred)

        # Update parameters dictionary
        params.update(grid_search.best_params_)

        # Log the parameters and best estimator
        self.mlflow_experiment(model_name, X_train, params, grid_search.best_estimator_, metrics)

    def train_logistic_regression(self, X_train, y_train, X_test, y_test):

        # Create the Logistic Regression model
        model = LogisticRegression(**self.config['LogisticRegression']['params'])

        # Train the model
        self.train_model('LogisticRegression', self.config['LogisticRegression']['params'], model,
                         X_train, y_train, X_test, y_test)

    def train_decision_tree(self, X_train, y_train, X_test, y_test):

        # Create the Decision Tree model
        model = DecisionTreeClassifier(**self.config['DecisionTree']['params'])

        # Train the model
        self.train_model('DecisionTree', self.config['DecisionTree']['params'], model,
                         X_train, y_train, X_test, y_test)

    def train_random_forest(self, X_train, y_train, X_test, y_test):

        # Create the Random Forest model
        model = RandomForestClassifier(**self.config['RandomForest']['params'])

        # Train the model
        self.train_model('RandomForest', self.config['RandomForest']['params'], model,
                         X_train, y_train, X_test, y_test)

    def train_xgboost(self, X_train, y_train, X_test, y_test):

        # Create the XGBoost model
        model = xgb.XGBClassifier(**self.config['XGBoost']['params'])

        # Train the model
        self.train_model('XGBoost', self.config['XGBoost']['params'], model,
                         X_train, y_train, X_test, y_test)

    def train_catboost(self, X_train, y_train, X_test, y_test):

        # Create the CatBoost model
        model = CatBoostClassifier(**self.config['CatBoost']['params'])

        # Train the model
        self.train_model('CatBoost', self.config['CatBoost']['params'], model,
                         X_train, y_train, X_test, y_test)

    def train_mlp(self, X_train, y_train, X_test, y_test):

        self.config['MLP']['params']['hidden_layer_sizes'] = tuple(self.config['MLP']['params']['hidden_layer_sizes'])

        # Create the MLP model
        model = MLPClassifier(**self.config['MLP']['params'])

        # Train the model
        self.train_model('MLP', self.config['MLP']['params'], model, X_train, y_train, X_test, y_test)

    def __call__(self, model):

        X_train, X_test, y_train, y_test = self.prepare_data()

        if model == 'lr':
            # Train Logistic Regression model
            self.train_logistic_regression(X_train, y_train, X_test, y_test)
        elif model == 'dt':
            # Train Decision Tree model
            self.train_decision_tree(X_train, y_train, X_test, y_test)
        elif model == 'rf':
            # Train Random Forest model
            self.train_random_forest(X_train, y_train, X_test, y_test)
        elif model == 'xgboost':
            # Train XGBoost model
            self.train_xgboost(X_train, y_train, X_test, y_test)
        elif model == 'catboost':
            # Train CatBoost model
            self.train_catboost(X_train, y_train, X_test, y_test)
        elif model == 'mlp':
            # Train MLP model
            self.train_mlp(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='lr',
                        choices=['lr', 'dt', 'rf', 'xgboost', 'catboost', 'mlp'])
    args = parser.parse_args()

    train_model = TrainModel()
    train_model(args.model)
    logger.info("Model training completed successfully.")
