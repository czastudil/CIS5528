import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    sampled = pd.read_csv("sampled.csv")
    unsampled = pd.read_csv("unsampled.csv")

    # Map color to the number of estimators
    n_est = [100, 150, 200, 250, 300]
    color_map = {100: "red", 150: "blue", 200: "cyan", 250: "green", 300: "orange"}

    # Create plot for max_depth, n_estimators, & accuracy
    sns.relplot(data=unsampled, x='max_depth', y='accuracy', hue='n_estimators', palette=color_map, hue_order=n_est, aspect=1.61)
    #plt.title('Accuracy for Model Trained on Full Feature Set')
    sns.relplot(data=sampled, x='max_depth', y='accuracy', hue='n_estimators', palette=color_map, hue_order=n_est, aspect=1.61)
    #plt.title('Accuracy for Model Trained on Sample of Feature Set')
    plt.show()
    