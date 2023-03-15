import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


def obtain_reduced_dataset(all_features, features_to_be_used):
    feature_names = ['air_time',
                     'disp_index',
                     'gmrt_in_air',
                     'gmrt_on_paper',
                     'max_x_extension',
                     'max_y_extension',
                     'mean_acc_in_air',
                     'mean_acc_on_paper',
                     'mean_gmrt',
                     'mean_jerk_in_air',
                     'mean_jerk_on_paper',
                     'mean_speed_in_air',
                     'mean_speed_on_paper',
                     'num_of_pendown',
                     'paper_time',
                     'pressure_mean',
                     'pressure_var',
                     'total_time']
    feature_names_to_be_used = []

    for feature_num in features_to_be_used:
        for feature_name in feature_names:
            feature_names_to_be_used.append(feature_name + feature_num)
    # print(feature_names_to_be_used)

    reduced_features = all_features.filter(feature_names_to_be_used)
    # print(reduced_features)
    return reduced_features


if __name__ == '__main__':
    # read in the dataset
    dataset = pd.read_csv("DARWIN.csv")
    # print(dataset)
    # print(dataset.info())

    # convert ids into number format
    dataset["ID"] = dataset["ID"].apply(lambda x: int(x.split("id_")[1]))

    # split into features (451 columns) and labels
    features = dataset.loc[:, dataset.columns != "class"]
    labels = dataset["class"]
    # print(features)
    # print(labels)

    # reduce features
    reduced_features_input = int(input("Would you like to reduce the features (0 for no, 1 for yes): "))
    reduced_features_to_be_used = ['17', '19', '21', '22', '23']
    if reduced_features_input:
        features = obtain_reduced_dataset(features, reduced_features_to_be_used)

    # split into train test
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    # print(x_train)
    # print(x_test)

    # create and train model
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)

    # get predictions
    y_pred = gnb.predict(x_test)
    # print(y_test.to_numpy())
    # print(y_pred)

    # get confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # calculate accuracy, specificity, & sensitivity
    accuracy = gnb.score(x_test, y_test)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    # report results
    if reduced_features_input:
        print("Results on Feature set (" + ','.join(reduced_features_to_be_used) + "): ")
    else:
        print("Results on Entire Feature set: ")
    print("     Naive Bayes accuracy: ", gnb.score(x_test, y_test))
    print("     Gaussian Naive Bayes specificity: ", specificity)
    print("     Gaussian Naive Bayes sensitivity: ", sensitivity)
