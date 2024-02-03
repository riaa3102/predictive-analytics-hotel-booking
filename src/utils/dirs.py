import os
from pathlib import Path


def get_parent_directory():
    return Path(__file__).resolve().parent.parent.parent


def create_directory(directory_path):
    directory_path.mkdir(parents=True, exist_ok=True)


def initialize_directories():
    parent_dir = get_parent_directory()
    src_dir = parent_dir / "src"
    data_dir = parent_dir / "data"
    raw_data_dir = data_dir / "raw"
    processed_data_dir = data_dir / "processed"
    experiments_dir = parent_dir / "experiments"
    mlexperiments_dir = experiments_dir / "mlflow"
    models_dir = parent_dir / "models"
    logs_dir = parent_dir / "logs"
    images_dir = parent_dir / "images"

    directories = {
        "parent_dir": parent_dir,
        "data_dir": data_dir,
        "raw_data_dir": raw_data_dir,
        "processed_data_dir": processed_data_dir,
        "experiments_dir": experiments_dir,
        "mlexperiments_dir": mlexperiments_dir,
        "mlruns_dir": mlexperiments_dir / "mlruns",
        "mlartifacts_dir": mlexperiments_dir / "mlartifacts",
        "models_dir": models_dir,
        "logs_dir": logs_dir,
        "images_dir": images_dir,
        "roc_plot_dir": images_dir / "roc_plot",
        "raw_data_file_path": raw_data_dir / "hotel-booking-demand" / "hotel_bookings.csv",
        "processed_data_file_path": processed_data_dir / "data.csv",
        "logs_file_path": logs_dir / "hotel_model.log",
        "config_file_path": src_dir / "config" / "config.yaml",
        "standard_scaler_file_path": models_dir / "standard_scaler.pkl",
        "one_hot_encoder_file_path": models_dir / "one_hot_encoder.pkl",
        "ordinal_encoder_file_path": models_dir / "ordinal_encoder.pkl",
        "target_encoder_file_path": models_dir / "target_encoder.pkl",
    }

    # Create directories (excluding keys ending with "_file_path")
    for key, directory_path in directories.items():
        if not key.endswith("_file_path"):
            create_directory(directory_path)

    return directories


# Initialize directories when the module is imported
DIRS = initialize_directories()

