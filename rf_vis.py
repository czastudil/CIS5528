import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    unsampled = pd.read_csv("full_feature_training.csv")
    sampled = pd.read_csv("reduced_feature_training.csv")

    # Map color to the number of estimators
    n_est = [100, 150, 200, 250, 300]
    color_map = {100: "red", 150: "blue", 200: "cyan", 250: "green", 300: "orange"}

    # Create plot for max_depth, n_estimators, & accuracy for sampled and unsampled features
    sns.relplot(data=unsampled, x='max_depth', y='mean_accuracy', hue='n_estimators', palette=color_map, hue_order=n_est, aspect=1.61)
    plt.title('Model Accuracy when Trained on Full Feature Set')
    sns.relplot(data=sampled, x='max_depth', y='mean_accuracy', hue='n_estimators', palette=color_map, hue_order=n_est, aspect=1.61)
    plt.title('Model Accuracy when Trained on Reduced Feture Set')
    plt.show()
    