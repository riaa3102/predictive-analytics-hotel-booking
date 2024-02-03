import os
import yaml
import pandas as pd
from src.utils.dirs import DIRS
from src.utils.logger import configure_logger


logger = configure_logger(name=os.path.basename(__file__)[:-3], log_level="INFO")


class DataPreprocessor:
    def __init__(self):
        with open(DIRS["config_file_path"], 'r') as file:
            self.config = yaml.safe_load(file)

    @staticmethod
    def handle_missing_values(df):
        # Fill NaN values in 'children' with 0
        df['children'] = df['children'].fillna(0)
        # Fill NaN values in 'country' with "unknown_value"
        df['country'] = df['country'].fillna("unknown_value")
        # Fill NaN values in 'agent' with 0
        df['agent'] = df['agent'].fillna(0)
        # Drop the 'company' column
        df = df.drop(columns='company')
        return df

    @staticmethod
    def change_data_type(df):
        df['children'] = df['children'].astype(int)
        df['agent'] = df['agent'].astype(int)
        return df

    @staticmethod
    def remove_duplicates(df):
        return df.drop_duplicates()

    @staticmethod
    def remove_leakage_features(df):
        exclude_columns = ['reservation_status', 'reservation_status_date', 'deposit_type']
        return df.drop(exclude_columns, axis=1)

    def feature_engineering(self, df):
        # Total Stay Length
        df['total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

        # Total Guests
        df['total_guests'] = df['adults'] + df['children'] + df['babies']

        # Room Type Change
        df['room_type_changed'] = (df['reserved_room_type'] != df['assigned_room_type']).astype(int)

        # Cancellation Rate
        df['cancellation_rate'] = df.apply(
            lambda row: row['previous_cancellations'] / (
                        row['previous_cancellations'] + row['previous_bookings_not_canceled'])
            if (row['previous_cancellations'] + row['previous_bookings_not_canceled']) > 0 else 0,
            axis=1
        )

        # Seasonality Feature (based on Northern Hemisphere)
        month_to_season = self.config['preprocess']['month_to_season']
        df['season'] = df['arrival_date_month'].map(month_to_season)

        # Weekday/Weekend of Arrival
        df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' +
                                            df['arrival_date_month'].astype(str) + '-' +
                                            df['arrival_date_day_of_month'].astype(str))

        df['is_weekend_arrival'] = (df['arrival_date'].dt.dayofweek >= 5).astype(int)
        df = df.drop(columns=['arrival_date'])

        # Special Request Count (as a categorical feature)
        df['special_requests_category'] = (pd.cut(df['total_of_special_requests'],
                                                  bins=[-1, 0, 2, float('inf')],
                                                  labels=[0, 1, 2])).astype(int)

        return df

    def __call__(self, df=None, save_csv=True):
        if df is None:
            df = pd.read_csv(DIRS["raw_data_file_path"])

        df = self.handle_missing_values(df)
        df = self.change_data_type(df)
        # df = self.remove_duplicates(df)
        df = self.remove_leakage_features(df)
        df = self.feature_engineering(df)

        if save_csv:
            df.to_csv(DIRS["processed_data_file_path"], index=False)
        else:
            return df


if __name__ == '__main__':
    data_preprocessor = DataPreprocessor()
    data_preprocessor()
    logger.info("Data preprocessing completed successfully.")
