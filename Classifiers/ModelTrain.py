import os
from typing import List

import joblib
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from utils.DefineData import *

TestRealLabels: List
TestPreLabels: List


def getTestRealLabels() -> List:
    return TestRealLabels


def getTestPreLabels() -> List:
    return TestPreLabels


def model_train(df, model_type, saved_model_path=SaveModelPath, trainedFeature: List[str] = None, maxdepth: int = 5):
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

    model = DecisionTreeClassifier(random_state=0, max_depth=maxdepth)
    if model_type == 'random_forest':
        # Numbers of decision trees is 100
        model = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=maxdepth)
    elif model_type == 'adaptive_boosting':
        # Numbers of decision trees is 100 and the maximum tree depth is 5
        estimator_cart = DecisionTreeClassifier(max_depth=maxdepth)
        model = AdaBoostClassifier(base_estimator=estimator_cart, n_estimators=100, random_state=0)

    # Split the data into train and test set
    if TIME_COLUMN_NAME in df.columns:
        df = df.drop(TIME_COLUMN_NAME, axis=1)
    x = df.drop(FAULT_FLAG, axis=1)
    y = df[FAULT_FLAG]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

    # Train the model
    model.fit(x_train, y_train)

    # 将所有的标签值按照顺序存起来
    alllabels = sorted(list(set(y)))
    alllabels = [str(i) for i in alllabels]
    with open("{}".format(os.path.join(saved_model_path, "alllabels.txt")), "w") as f:
        f.write("\n".join(alllabels))


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
