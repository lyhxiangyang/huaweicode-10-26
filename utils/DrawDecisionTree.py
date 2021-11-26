import os.path

import matplotlib.pyplot as plt
import sklearn
from sklearn import tree
import joblib

from utils.DefineData import MODEL_TYPE


def plot_tree_structure(depth, modelpath: str, filename: str = None):
    if filename is None:
        filename = MODEL_TYPE[0] + ".pkl"
    # Load the model
    modelname = os.path.join(modelpath, filename)
    model = joblib.load(modelname)
    # 读取标签信息
    with open("{}".format(os.path.join(modelpath, "alllabels.txt")), "r") as f:
        classes = f.read().splitlines()
    # 读取头文件信息
    with open("{}".format(os.path.join(modelpath, "header.txt")), "r") as f:
        features = f.read().splitlines()

    plt.figure(dpi=300, figsize=(24, 10))
    tree.plot_tree(model, max_depth=depth, feature_names=features, class_names=classes, filled=True)
    plt.show()
