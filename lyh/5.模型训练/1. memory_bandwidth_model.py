import os

# 主要是使用topdowns数据中的 训练处一个模型出来
# 主要是使用自己构建出来的已经归一化出来的topdown数据进行训练
import sys

import pandas as pd

from hpc.classifiers.ModelTrain import model_train
from hpc.l3l2utils.DataFrameSaveRead import getfilepd
from hpc.l3l2utils.DefineData import MODEL_TYPE

if __name__ == "__main__":
    # 需要用到的三个路径
    nowpath = sys.path[0]
    savemodelpath = os.path.join(nowpath, "models", "grapes")
    traindatapath = os.path.join(nowpath, "model_grape_50_110.csv")
    # trainFeatures = ["ddrc_ddwr_sum"]
    trainFeatures = ["pgfree_mean",]

    # ============================================================= 对训练数据进行读取
    trainpd = getfilepd(ipath=traindatapath)

    # =============================================================
    model_train(trainpd, MODEL_TYPE[0], saved_model_path=savemodelpath, trainedFeature=trainFeatures, maxdepth=2)

