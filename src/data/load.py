import os
import opendatasets as od
from src.utils.dirs import DIRS
from src.utils.logger import configure_logger

logger = configure_logger(name=os.path.basename(__file__)[:-3], log_level="INFO")


class DataLoader:
    def __init__(self):
        pass

    @staticmethod
    def download_data():
        dataset_url = "https://www.kaggle.com/jessemostipak/hotel-booking-demand"
        od.download(dataset_url, DIRS["raw_data_dir"])

    def __call__(self, *args, **kwargs):
        self.download_data()


if __name__ == "__main__":
    data_loader = DataLoader()
    data_loader()
    logger.info("Data loading completed successfully.")
