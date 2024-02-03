import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from src.utils.dirs import DIRS


class DataVisualizer:
    def __init__(self):
        with open(DIRS["config_file_path"], 'r') as file:
            self.config = yaml.safe_load(file)

        self.roc_plot_dir = DIRS["roc_plot_dir"]

    def plot_roc_curve(self, model_name, y_test, y_pred):
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))

        plt.plot(fpr, tpr, lw=2, label=self.config['transform']['target_column'])

        plt.plot([0, 1], [0, 1], lw=2, linestyle='--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Postive Rate')
        plt.ylabel('True Positive Rate')

        plt.title(f'{model_name} ROC curve (AUC = {roc_auc:.2f})')

        plt.legend(loc='lower right')

        plt.grid(True)

        plt.savefig(self.roc_plot_dir / f"{model_name}_ROC_curve")
        # plt.show()
