import os.path
from typing import Dict

import joblib
import pandas as pd

from hpc.l3l2utils.DataFrameOperation import smoothseries, getSeriesFrequencyMean
from hpc.l3l2utils.DataFrameSaveRead import savepdfile
from hpc.l3l2utils.DataOperation import getsametimepdList
from hpc.l3l2utils.DefineData import MODEL_TYPE, TIME_COLUMN_NAME, FAULT_FLAG
from hpc.l3l2utils.ModelTrainLib import getAllDataFramesFromDectionJson
from hpc.l3l2utils.ParsingJson import readJsonToDict, saveDictToJson

"""
从正常文件中获得数据的平均值
"""


def changeNormalData(inputDict: Dict) -> Dict:
    normalDataJson = readJsonToDict(*(os.path.split(inputDict["normalpath"])))
    alldatapdDicts = getAllDataFramesFromDectionJson(normalDataJson)
    debugpd = pd.DataFrame()
    alldatapdDicts["server"], alldatapdDicts["topdown"] = getsametimepdList(
        [alldatapdDicts["server"], alldatapdDicts["topdown"]])
    debugpd[TIME_COLUMN_NAME] = alldatapdDicts["server"][TIME_COLUMN_NAME]
    debugpd[FAULT_FLAG] = alldatapdDicts["server"][FAULT_FLAG]
    serverfeatures = ["pgfree"]
    for ife in serverfeatures:
        alldatapdDicts["server"][ife] = smoothseries(alldatapdDicts["server"][ife])
        inputDict["normalDataMean"]["server"][ife] = getSeriesFrequencyMean(alldatapdDicts["server"][ife])
        debugpd["server_{}".format(ife)] = alldatapdDicts["server"][ife]
        debugpd["server_{}_mean".format(ife)] = inputDict["normalDataMean"]["server"][ife]
    topdwonfeatures = ["mflops", "ddrc_rd", "ddrc_wr"]
    for ife in topdwonfeatures:
        alldatapdDicts["topdown"][ife] = smoothseries(alldatapdDicts["topdown"][ife])
        inputDict["normalDataMean"]["topdown"][ife] = getSeriesFrequencyMean(alldatapdDicts["topdown"][ife])
        debugpd["topdown_{}".format(ife)] = alldatapdDicts["topdown"][ife]
        debugpd["topdown_{}_mean".format(ife)] = inputDict["normalDataMean"]["topdown"][ife]

    if inputDict["spath"] is not None:
        tpath = os.path.join(inputDict["spath"], "meanvalues")
        saveDictToJson(inputDict, tpath, "config_change.json")
        savepdfile(debugpd, tpath, "feature_meanvalues.csv")


# 修改阈值
"""
1. 修改阈值
2. 修改平均值
"""


def preproccessConfigfile(inputDict: Dict) -> Dict:
    # 对阈值重新进行设置，必须为决策树
    def changeFirstThread(modelpath: str, threads):
        f = open(modelpath, 'rb')
        model = joblib.load(f)
        f.close()
        model.tree_.threshold[0] = threads
        joblib.dump(model, modelpath)

    # 1. 预处理第一步对内存泄露的模型进行重新设置
    if inputDict["memleakpermin"] is not None:
        modelpath = os.path.join(inputDict["servermemory_modelpath"],
                                 MODEL_TYPE[inputDict["servermemory_modeltype"]] + ".pkl")
        modelthread = inputDict["memleakpermin"]
        changeFirstThread(modelpath, modelthread)
    # 2. 对内存带宽模型的阈值进行设置
    if inputDict["pgfree_thread"] is not None:
        modelpath = os.path.join(inputDict["serverbandwidth_modelpath"],
                                 MODEL_TYPE[inputDict["serverbandwidth_modeltype"]] + ".pkl")
        modelthread = inputDict["pgfree_thread"]
        changeFirstThread(modelpath, modelthread)
    if inputDict["ddrc_ddwr_sum_max"] is not None:
        modelpath = os.path.join(inputDict["cachegrab_modelpath"],
                                 MODEL_TYPE[inputDict["cachegrab_modelpath"]] + ".pkl")
        cachethread = inputDict["ddrc_ddwr_sum_max"]
        changeFirstThread(modelpath, cachethread)

    # 对文件中的各个平均值进行处理 pgfree mflops ddrc_rd ddrc_wr
    if inputDict["spath"] is not None:
        changeNormalData(inputDict)
