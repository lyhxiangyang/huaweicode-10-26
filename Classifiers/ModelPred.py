import os
from collections import defaultdict
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

# 返回一个字典结构 异常-List[准确率]
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
    # 得到所有的lables
    classes = [0]
    with open("{}".format(os.path.join(saved_model_path, "alllabels.txt")), "r") as f:
        classes = f.read().splitlines()
        classes = [int(i) for i in classes]
    y_pred_probability = model_pred_probability(df_selected, model_type, saved_model_path=saved_model_path)
    classes_probability_Dict = defaultdict(list)
    for iline in y_pred_probability:
        for icolumns, iclass in enumerate(classes):
            classes_probability_Dict[iclass].append(iline[icolumns])
    return classes_probability_Dict

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


"""
识别温度数据
"""
def predictTemp(model_path: str, model_type: str, data: pd.DataFrame):
    FANS = [
        'FAN1_F_Speed', "FAN1_R_Speed",
        'FAN2_F_Speed', "FAN2_R_Speed",
        'FAN3_F_Speed', "FAN3_R_Speed",
        'FAN4_F_Speed', "FAN4_R_Speed",
        'FAN5_F_Speed', "FAN5_R_Speed",
        'FAN6_F_Speed', "FAN6_R_Speed",
        'FAN7_F_Speed', "FAN7_R_Speed",
    ]
    TEMPERATURE = [
        'CPU1_Core_Rem', 'CPU2_Core_Rem', 'CPU3_Core_Rem', 'CPU4_Core_Rem',
        'CPU1_MEM_Temp', 'CPU2_MEM_Temp', 'CPU3_MEM_Temp', 'CPU4_MEM_Temp',
    ]
    def get_extended_features(prefix):
        selected = []
        for p in prefix:
            selected.append(p)
            selected.append(p + '_max')
            selected.append(p + '_min')
            selected.append(p + '_mean')
            selected.append(p + '_percentage50')
        return selected
    result = []
    for i, temp in enumerate(TEMPERATURE):
        for j, fan in enumerate(FANS):
            extended_features = get_extended_features(['freq', temp, fan])
            select_data = data[extended_features]
            model = joblib.load(os.path.join(model_path, model_type + '.pkl'))
            y = model.predict(select_data)
            if i == 0 and j == 0:
                result = y
            for k, v in enumerate(y):
                if v == 3:
                    if result[k] == 0:
                        result[k] = 3
                if v == 4:
                    if result[k] == 0 or result[k] == 3:
                        result[k] = 4
    return result


















































































































