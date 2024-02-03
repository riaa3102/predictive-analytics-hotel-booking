import os
import numpy as np
from typing import Optional
import yaml
from pickle import load, dump
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder
from src.utils.dirs import DIRS
from src.utils.logger import configure_logger


logger = configure_logger(name=os.path.basename(__file__)[:-3], log_level="INFO")


class DataTransformer:
    def __init__(self, load_flag: bool = False):

        with open(DIRS["config_file_path"], 'r') as file:
            self.config = yaml.safe_load(file)

        self.standard_scaler_columns = self.config['transform']['standard_scaler_columns']
        self.one_hot_encoder_columns = self.config['transform']['one_hot_encoder_columns']
        self.ordinal_encoder_columns = self.config['transform']['ordinal_encoder_columns']
        self.target_encoder_columns = self.config['transform']['target_encoder_columns']

        self.standard_scaler = None
        self.one_hot_encoder = None
        self.ordinal_encoder = None
        self.target_encoder = None

        if load_flag:
            self.load_scaler()
            self.load_encoder()
        else:
            self.standard_scaler = StandardScaler()
            self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.int32)
            self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int32,
                                                  categories=[
                                                      self.config['transform']['ordinal_encoder_categories'].get(col,
                                                                                                                 None)
                                                      for col in self.ordinal_encoder_columns])
            self.target_encoder = TargetEncoder()

    def save_scaler(self):
        dump(self.standard_scaler, open(DIRS["standard_scaler_file_path"], 'wb'))
        logger.info("Standard Scaler saved successfully.")

    def save_encoder(self):
        dump(self.one_hot_encoder, open(DIRS["one_hot_encoder_file_path"], 'wb'))
        logger.info("One-Hot Encoder saved successfully.")
        dump(self.ordinal_encoder, open(DIRS["ordinal_encoder_file_path"], 'wb'))
        logger.info("Ordinal Encoder saved successfully.")
        dump(self.target_encoder, open(DIRS["target_encoder_file_path"], 'wb'))
        logger.info("Target Encoder saved successfully.")

    def load_scaler(self):
        self.standard_scaler = load(open(DIRS["standard_scaler_file_path"], 'rb'))
        logger.info("Standard Scaler loaded successfully.")

    def load_encoder(self):
        self.one_hot_encoder = load(open(DIRS["one_hot_encoder_file_path"], 'rb'))
        logger.info("One-Hot Encoder loaded successfully.")
        self.ordinal_encoder = load(open(DIRS["ordinal_encoder_file_path"], 'rb'))
        logger.info("Ordinal Encoder loaded successfully.")
        self.target_encoder = load(open(DIRS["target_encoder_file_path"], 'rb'))
        logger.info("Target Encoder loaded successfully.")

    def scaler(self, X_train: Optional, X_test: Optional):

        if X_train is not None:
            X_train[self.standard_scaler_columns] = self.standard_scaler.fit_transform(X_train[self.standard_scaler_columns])
            self.save_scaler()
        if X_test is not None:
            X_test[self.standard_scaler_columns] = self.standard_scaler.transform(X_test[self.standard_scaler_columns])

        return X_train, X_test

    def encoder(self, X_train: Optional, X_test: Optional, y_train: Optional):

        if X_train is not None:
            X_train_encoded = self.one_hot_encoder.fit_transform(X_train[self.one_hot_encoder_columns])
            X_train = X_train.drop(columns=self.one_hot_encoder_columns)
            X_train[self.one_hot_encoder.get_feature_names_out(self.one_hot_encoder_columns)] = X_train_encoded
            X_train[self.ordinal_encoder_columns] = self.ordinal_encoder.fit_transform(X_train[self.ordinal_encoder_columns])
            X_train[self.target_encoder_columns] = self.target_encoder.fit_transform(X_train[self.target_encoder_columns], y_train)
            self.save_encoder()
        if X_test is not None:
            X_test_encoded = self.one_hot_encoder.transform(X_test[self.one_hot_encoder_columns])
            X_test = X_test.drop(columns=self.one_hot_encoder_columns)
            X_test[self.one_hot_encoder.get_feature_names_out(self.one_hot_encoder_columns)] = X_test_encoded
            X_test[self.ordinal_encoder_columns] = self.ordinal_encoder.transform(X_test[self.ordinal_encoder_columns])
            X_test[self.target_encoder_columns] = self.target_encoder.transform(X_test[self.target_encoder_columns])

        return X_train, X_test

    def __call__(self, X_train: Optional = None, X_test: Optional = None, y_train: Optional = None):
        X_train, X_test = self.scaler(X_train, X_test)
        X_train, X_test = self.encoder(X_train, X_test, y_train)
        return X_train, X_test


