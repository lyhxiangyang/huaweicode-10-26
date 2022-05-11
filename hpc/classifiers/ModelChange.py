# 使用三种模型进行预测信息
import os

import joblib
import matplotlib.pyplot as plt
from sklearn import tree

from hpc.l3l2utils.DefineData import MODEL_TYPE


def show_threshold(filepath):
    f = open(filepath, 'rb')
    model = joblib.load(f)
    f.close()
    print('node features  :', model.tree_.feature)
    print('node thresholds:', model.tree_.threshold)


def change_threshold(filepath, node, value):
    f = open(filepath, 'rb')
    model = joblib.load(f)
    f.close()
    model.tree_.threshold[node] = value
    joblib.dump(model, filepath)
    # print('features and thresholds after modification:')
    # show_threshold(filepath)

# if __name__ == '__main__':
#     file = r'hjx\decision_tree.pkl'
#     show_threshold(file)
#     change_threshold(file, 0, 106)

def plot_tree_structure(depth: int = None, modelpath: str = "", filename: str = None):
    if filename is None:
        filename = MODEL_TYPE[0] + ".pkl"
    # Load the model
    modelname = os.path.join(modelpath, filename)
    model = joblib.load(modelname)
    if depth is None:
        depth = model.max_depth
    # 读取标签信息
    with open("{}".format(os.path.join(modelpath, "alllabels.txt")), "r") as f:
        classes = f.read().splitlines()
    print("classes: {}".format(classes))
    # 读取头文件信息
    with open("{}".format(os.path.join(modelpath, "header.txt")), "r") as f:
        features = f.read().splitlines()
    print("features: {}".format(features))

    plt.figure(dpi=300, figsize=(24, 10))
    tree.plot_tree(model, max_depth=depth, feature_names=features, class_names=classes, filled=True)
    plt.show()
