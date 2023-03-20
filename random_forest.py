import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

    reduced_features = all_features.filter(feature_names_to_be_used)
    return reduced_features

def get_average_performance(iterations, features, labels):
    accuracy_sum = 0
    sensitivity_sum = 0
    specificity_sum = 0
    for _ in range(iterations):
        # split into train test
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        #x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

        # create and train model
        n_est = 200
        m_depth = 6
        forest = RandomForestClassifier(n_estimators=n_est, max_depth=m_depth, bootstrap=True, min_samples_split=2, min_samples_leaf=1)
        forest.fit(x_train, y_train)
        
        # get predictions
        y_pred = forest.predict(x_test)

        # get confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # calculate accuracy, specificity, & sensitivity
        accuracy_sum += forest.score(x_test, y_test)
        specificity_sum += tn / (tn + fp)
        sensitivity_sum += tp / (tp + fn)

    reduced_features_to_be_used = ['17', '19', '21', '22', '23']
    print("Iterations:", iterations)
    print("max_depth:", m_depth)
    print("n_estimators:", n_est)
    if reduced_features_input:
        print("Results on Feature set (" + ','.join(reduced_features_to_be_used) + "): ")
    else:
        print("Results on Entire Feature set: ")
    print("     Random Forest accuracy: ", accuracy_sum/iterations)
    print("     Random Forest specificity: ", specificity_sum/iterations)
    print("     Random Forest sensitivity: ", sensitivity_sum/iterations)
    

if __name__ == '__main__':
    # read in the dataset
    dataset = pd.read_csv("DARWIN.csv")

    # convert ids into number format
    dataset["ID"] = dataset["ID"].apply(lambda x: int(x.split("id_")[1]))

    # split into features (451 columns) and labels
    features = dataset.loc[:, dataset.columns != "class"]
    labels = dataset["class"]

    # reduce features
    reduced_features_input = int(input("Would you like to reduce the features (0 for no, 1 for yes): "))
    reduced_features_to_be_used = ['17', '19', '21', '22', '23']
    if reduced_features_input:
        features = obtain_reduced_dataset(features, reduced_features_to_be_used)

    get_average_performance(10, features, labels)
    """# split into train test
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    #x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    # create and train model
    forest = RandomForestClassifier(n_estimators=100, max_depth=4, bootstrap=True, min_samples_split=2, min_samples_leaf=1)
    forest.fit(x_train, y_train)
    
    # get predictions
    y_pred = forest.predict(x_test)

    # get confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # calculate accuracy, specificity, & sensitivity
    accuracy = forest.score(x_test, y_test)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    # report results
    if reduced_features_input:
        print("Results on Feature set (" + ','.join(reduced_features_to_be_used) + "): ")
    else:
        print("Results on Entire Feature set: ")
    print("     Random Forest accuracy: ", forest.score(x_test, y_test))
    print("     Random Forest specificity: ", specificity)
    print("     Random Forest sensitivity: ", sensitivity)"""