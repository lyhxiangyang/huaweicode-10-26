import os
from typing import Dict

import joblib
import pandas as pd

from utils.DefineData import SaveModelPath, MODEL_TYPE


def model_pred(x_pred, model_type, saved_model_path=SaveModelPath):
    """
    Use trained model to predict
    :param saved_model_path:
    :param x_pred: Samples to be predicted
    :param model_type: The type of saved model to be used
    :return y_pred: Class labels for samples in x_pred
    """
    # Load saved model
    model = joblib.load('%s\\%s.pkl' % (saved_model_path, model_type))

    y_pred = model.predict(x_pred)
    return y_pred

"""
判断预测的概率
"""
def model_pred_probability(x_pred, model_type, saved_model_path=SaveModelPath):
    model = joblib.load('%s\\%s.pkl' % (saved_model_path, model_type))

    y_pred = model.predict_proba(x_pred)
    return y_pred

def select_and_pred_probability(df, model_type, saved_model_path=SaveModelPath):
    # 读取头文件信息
    with open("{}".format(os.path.join(saved_model_path, "header.txt")), "r") as f:
        features = f.read().splitlines()
    # 判断有哪些特征值
    columns = set(df.columns.array)
    fessets = set(features)
    if not fessets.issubset(columns):
        print("预测过程中选择特征不存在")
        print("=================不存在特征如下")
        print(fessets - columns)
        print("=================")
        exit(1)
    df_selected = df[features]
    # Use trained model to predict
    y_pred = model_pred_probability(df_selected, model_type, saved_model_path=saved_model_path)
    return y_pred

"""
进行预测
"""
def select_and_pred(df, model_type, saved_model_path=SaveModelPath):
    # 读取头文件信息
    with open("{}".format(os.path.join(saved_model_path, "header.txt")), "r") as f:
        features = f.read().splitlines()
    # 判断有哪些特征值
    columns = set(df.columns.array)
    fessets = set(features)
    if not fessets.issubset(columns):
        print("预测过程中选择特征不存在")
        print("=================不存在特征如下")
        print(fessets - columns)
        print("=================")
        exit(1)
    df_selected = df[features]
    # Use trained model to predict
    y_pred = model_pred(df_selected, model_type, saved_model_path=saved_model_path)
    return y_pred


"""
将每个核心上的数据都进行预测
"""


def predictFilename_Time_Core(ftcPD: Dict, modelpath: str):
    filename_time_corePd = {}
    for filename, time_core_pdDict in ftcPD.items():
        filename_time_corePd[filename] = {}
        for time, core_pdDict in time_core_pdDict.items():
            filename_time_corePd[filename][time] = {}
            for icore, tpd in core_pdDict.items():
                tpd: pd.DataFrame
                print("{}-{}-{}".format(filename, time, icore))
                for itype in MODEL_TYPE:
                    prelist = select_and_pred(tpd, model_type=itype, saved_model_path=modelpath)
                    tpd[itype + "_flag"] = prelist
