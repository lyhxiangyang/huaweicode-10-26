import os

# 主要是使用topdowns数据中的 训练处一个模型出来
# 主要是使用自己构建出来的已经归一化出来的topdown数据进行训练
import sys

import joblib
from sklearn.tree import DecisionTreeClassifier

def change_threshold(filepath, node, value):
    f = open(filepath, 'rb')
    model = joblib.load(f)
    f.close()
    model.tree_.threshold[node] = value
    joblib.dump(model, filepath)
    print('features and thresholds after modification:')


if __name__ == "__main__":
    # 需要用到的三个路径
    nowpath = sys.path[0]
    savemodelpath = os.path.join(nowpath, "models", "decision_tree.pkl")
    # ============================================================= 对模型数据进行分析
    f = open(savemodelpath, 'rb')
    model = joblib.load(f)
    f.close()
    model: DecisionTreeClassifier
    # ============================================================= 对模型数据进行分析
    tree_ = model.tree_
    tree_.threshold[0] = 6000
    print("threshold: {}".format(tree_.threshold))
    print("feature: {}".format(tree_.feature))

    print("tree_left: {}".format(tree_.children_left)) # 前序遍历
    print("tree_right: {}".format(tree_.children_right))
    # ============================================================= 保存到model1
    joblib.dump(model, savemodelpath)


