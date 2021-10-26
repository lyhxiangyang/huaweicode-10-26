import os.path

import matplotlib.pyplot as plt
from sklearn import tree
import joblib

from utils.DefineData import MODEL_TYPE


def plot_tree_structure(depth, modelpath: str):
    # Load the model
    modelname = os.path.join(modelpath, MODEL_TYPE[0] + ".pkl")
    model = joblib.load(modelname)
    classes = ["Normal", "CPU_ALL"]
    headerfile = os.path.join(modelpath, "header.txt")
    f = open(headerfile, 'r')

    # Read the features
    features = f.read().splitlines()

    plt.figure(dpi=300, figsize=(24, 10))
    tree.plot_tree(model, max_depth=depth, feature_names=features, class_names=classes, filled=True)
    plt.show()
