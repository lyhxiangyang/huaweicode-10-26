from typing import List

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
import joblib
from utils.DefineData import *
import os

TestRealLabels: List
TestPreLabels: List


def getTestRealLabels() -> List:
    return TestRealLabels


def getTestPreLabels() -> List:
    return TestPreLabels


def model_train(df, model_type, saved_model_path=SaveModelPath, trainedFeature: List[str] = None):
    """
    Train the model of selected type
    :param saved_model_path:
    :param trainedFeature:
    :param df: Dataframe of selected features and labels
    :param model_type: The type of model to be trained
    """
    # Remove column "Intensity" if exists
    if trainedFeature is not None:
        if FAULT_FLAG not in trainedFeature:
            trainedFeature.append(FAULT_FLAG)
        if TIME_COLUMN_NAME not in trainedFeature:
            trainedFeature.append(TIME_COLUMN_NAME)
        df = df[trainedFeature]

    # 如果有Intensity这个 就使用Intensity
    # if header.count('Intensity'):
    #     header.remove('Intensity')
    # df = df[header]

    model = DecisionTreeClassifier(random_state=0)
    if model_type == 'random_forest':
        # Numbers of decision trees is 100
        model = RandomForestClassifier(n_estimators=100, random_state=0)
    elif model_type == 'adaptive_boosting':
        # Numbers of decision trees is 100 and the maximum tree depth is 5
        estimator_cart = DecisionTreeClassifier(max_depth=5)
        model = AdaBoostClassifier(base_estimator=estimator_cart, n_estimators=100, random_state=0)

    # Split the data into train and test set
    if TIME_COLUMN_NAME in df.columns:
        df = df.drop(TIME_COLUMN_NAME, axis=1)
    x = df.drop(FAULT_FLAG, axis=1)
    y = df[FAULT_FLAG]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

    # Train the model
    model.fit(x_train, y_train)

    # Test the accuracy of the model
    y_eval = model.predict(x_test)

    global TestRealLabels
    global TestPreLabels
    TestRealLabels = list(y_test)
    TestPreLabels = list(y_eval)

    accuracy = metrics.accuracy_score(y_test, y_eval)

    if not os.path.exists(saved_model_path):
        os.makedirs(saved_model_path)

    # Save model
    joblib.dump(model, '%s\\%s.pkl' % (saved_model_path, model_type))

    # Save the header without label
    header = list(df.columns)
    header.remove(FAULT_FLAG)
    f = open('%s\\header.txt' % saved_model_path, 'w')
    f.write('\n'.join(header))
    f.close()
    return accuracy
