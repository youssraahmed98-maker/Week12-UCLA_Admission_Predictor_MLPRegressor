import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_correlation_heatmap(data):
    """
    Plot a correlation heatmap for the given data.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred):
    """
    Plot actual values against predicted values for a regression model.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred)

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')

    plt.xlabel('Actual Admit Chance', fontsize=12)
    plt.ylabel('Predicted Admit Chance', fontsize=12)
    plt.title('Actual vs Predicted', fontsize=16)
    plt.tight_layout()
    plt.show()