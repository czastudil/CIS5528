import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import csv
# from tree_vis import *
import random
from tqdm import tqdm


def get_average_performance(features, labels, writer):
    est_list = [100, 150, 200, 250, 300]
    depth_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    max_acc = -1.0
    best_est = None

    for x in tqdm(depth_list):
        for y in tqdm(est_list):
            # split into train test
            x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
            forest = RandomForestClassifier(n_estimators=y, max_depth=x, bootstrap=False, min_samples_split=3,
                                            max_leaf_nodes=5, random_state=22)
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
    dft = pd.read_csv('archive/data_augmented.csv')

    print("Loaded in Dataset ...")

    # preprocess data
    dft = dft.drop(dft[dft['VSx'] == '39/ 55'].index)
    dft = dft.drop(dft[dft['VSx'] == '209 /224'].index)
    dft = dft.astype('float')

    # split into features (451 columns) and labels
    column_headers = ['VHD', 'VLV', 'VmC', 'VE', 'VSx', 'VL', 'Men', 'Femal', 'Age']
    # column_headers = ['VHD', 'VLV', 'VmC', 'VE', 'VSx', 'VL', 'Age']
    features = dft[column_headers]
    labels = dft['Label']

    print("Preprocessing Done ...")

    model = None

    f = open('feature_training.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['max_depth', 'n_estimators', 'accuracy', 'specificity', 'sensitivity'])

    print("Calculating Performance ...")

    model = get_average_performance(features, labels, writer)
    f.close()

    results_file = pd.read_csv('feature_training.csv')
    accuracy_list = results_file['accuracy']

    accuracy_sum = 0
    for accuracy in accuracy_list:
        accuracy_sum += accuracy
    average_accuracy = accuracy_sum / len(accuracy_list)

    print("Accuracy: " + str(average_accuracy))

    sample_tree_num = random.randint(0, model.get_params()['n_estimators'])
    # vis_tree(model.estimators_[sample_tree_num],
    #          construct_feature_names(reduced_features_input, reduced_features_to_be_used),
    #          ['P', 'H'], 'tree_sampled')
