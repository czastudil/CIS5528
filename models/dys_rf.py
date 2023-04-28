import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import csv
import matplotlib.pyplot as plt
from tree_vis import *

feature_names = ['age',
                 'gender',
                 'laterality',
                 'BHK_raw_speed_score',
                 'BHK_raw_quality_score',
                 'median_Freq_speed',
                 'dist_Freq_speed',
                 'in_Air,Space_Between_Words',
                 'std_Density',
                 'median_Freq_tremolo',
                 'mean_d_P',
                 'std_d_P',
                 'mean_Pressure',
                 'dist_Freq_tilt_x',
                 'bandwidth_tilt_x',
                 'median_Freq_tilt_y']

def get_performance(features, labels, writer):
    est_list = [100, 150, 200, 250, 300]
    depth_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    max_acc = -1.0
    best_est = None

    for x in depth_list:
        for y in est_list:
            # split into train test
            x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
            forest = RandomForestClassifier(n_estimators=y, max_depth=x, bootstrap=False, min_samples_split=2, min_samples_leaf=1)
            forest.fit(x_train, y_train)
            c_val_score = np.mean(cross_val_score(forest, x_train, y_train, cv=10))
                
            # get predictions
            y_pred = forest.predict(x_test)

            # get confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            # calculate accuracy, specificity, & sensitivity
            accuracy = c_val_score
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)

            if accuracy > max_acc:
                best_est = forest
            
            writer.writerow([x, y, accuracy, specificity, sensitivity])
    return best_est

if __name__ == '__main__':
    # read in the dataset
    dataset = pd.read_csv("dysgraphia_typical_dataset.csv")
    dataset['group'] = dataset['group'].astype('category')
    dataset['gender'] = dataset['gender'].astype('category')
    dataset['laterality'] = dataset['laterality'].astype('category')

    # convert group, gender, laterality to numerical values
    cat_columns = dataset.select_dtypes(['category']).columns
    dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)

    # convert age to numerical value
    dataset['age'] = dataset['age'].apply(lambda x: float(x.split("ans")[0]) + float(x.split("ans")[1])/12 )

    # shuffle the dataset
    dataset = dataset.sample(frac=1)

    # split into features (451 columns) and labels
    features = dataset.loc[:, dataset.columns != "group"]
    labels = dataset["group"]

    model = None
    # create csv to save training progression data
    f = open('dysgraphia_training.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['max_depth','n_estimators','accuracy','specificity','sensitivity'])
    model = get_performance(features, labels, writer)
    importances =  model.feature_importances_
    sorted_indices = model.feature_importances_.argsort()
    plt.title('Feature Importance')
    print('INDICES:', sorted_indices)
    print('FEATURES:', features.columns)
    feat_names = [features.columns[i] for i in sorted_indices]
    plt.bar(feat_names, model.feature_importances_[sorted_indices])
    plt.xticks(rotation=90)
    plt.show()
    f.close()