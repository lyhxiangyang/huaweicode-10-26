# 使用三种模型进行预测信息
import os
from typing import List

import pandas as pd

from Classifiers.ModelPred import select_and_pred
from Classifiers.ModelTrain import model_train, getTestRealLabels, getTestPreLabels
from utils.DataFrameOperation import PushLabelToFirst, PushLabelToEnd
from utils.DefineData import MODEL_TYPE, FAULT_FLAG, TIME_COLUMN_NAME
from utils.GetMetrics import get_metrics


def TrainThree(trainedpd: pd.DataFrame, spath: str, modelpath: str = "Classifiers/saved_model/tmp",
               selectedFeature: List[str] = None):
    if os.path.exists(spath):
        pass
    else:
        os.makedirs(spath)
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)
    tDic = {}
    for itype in MODEL_TYPE:
        accuracy = model_train(trainedpd, itype, saved_model_path=modelpath, trainedFeature=selectedFeature)
        print('Accuracy of %s classifier: %f' % (itype, accuracy))
        tallFault = sorted(list(set(getTestRealLabels())))
        for i in tallFault:
            if i not in tDic.keys():
                tDic[i] = {}
            tmetrics = get_metrics(getTestRealLabels(), getTestPreLabels(), label=i)

            tDic[i]["accuracy_" + itype] = tmetrics["accuracy"]
            tDic[i]["precision_" + itype] = tmetrics["precision"]
            tDic[i]["recall_" + itype] = tmetrics["recall"]

    itpd = pd.DataFrame(data=tDic).T
    savefilename = "1.模型训练过程中数据统计.csv"
    itpd.to_csv(os.path.join(spath, savefilename))


def testThree(testpd: pd.DataFrame, spath: str, modelpath: str = "Classifiers/saved_model/tmp"):
    if not os.path.exists(spath):
        os.makedirs(spath)
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)
    tDic = {}
    reallist = testpd[FAULT_FLAG]
    resDict = {}
    resDict[FAULT_FLAG] = list(reallist)
    resDict[TIME_COLUMN_NAME] = list(testpd[TIME_COLUMN_NAME])
    for itype in MODEL_TYPE:
        prelist = select_and_pred(testpd, model_type=itype, saved_model_path=modelpath)
        resDict[itype + "_labels"] = prelist
        anumber = len(prelist)
        rightnumber = len([i for i in range(0, len(prelist)) if prelist[i] == reallist[i]])
        print("{}: 一共预测{}数据，其中预测正确{}数量, 正确率{}".format(itype, anumber, rightnumber, rightnumber / anumber))
        tallFault = sorted(list(set(reallist)))
        for i in tallFault:
            if i not in tDic.keys():
                tDic[i] = {}
            tmetrics = get_metrics(reallist, prelist, i)
            if "num" not in tDic[i].keys():
                tDic[i]["num"] = tmetrics["realnums"][i]
            # 将数据进行保存
            tDic[i]["accuracy_" + itype] = tmetrics["accuracy"]
            tDic[i]["precision_" + itype] = tmetrics["precision"]
            tDic[i]["recall_" + itype] = tmetrics["recall"]
            tDic[i]["per_itself_" + itype] = tmetrics["per_itself"]
            tDic[i]["per_normal_" + itype] = tmetrics["per_normal"]
            tDic[i]["per_samefault_" + itype] = tmetrics["per_samefault"]
            tDic[i]["per_fault_" + itype] = tmetrics["per_fault"]
            print("{}-fault:{}-per_normal:{} + per_fault:{} == {} 理论应该等1".format(itype, i, tmetrics["per_normal"], tmetrics["per_fault"], tmetrics["per_normal"] + tmetrics["per_fault"]))

    itpd = pd.DataFrame(data=tDic).T
    savefilename = "1.预测数据信息统计.csv"
    itpd.to_csv(os.path.join(spath, savefilename), index=True)

    ittpd = pd.DataFrame(data=resDict)
    ittpd = PushLabelToFirst(ittpd, TIME_COLUMN_NAME)
    ittpd = PushLabelToEnd(ittpd, FAULT_FLAG)
    savefilename = "2.三种模型预测值比较.csv"
    ittpd.to_csv(os.path.join(spath, savefilename), index=False)


    print("预测信息结束")


"""
中间文件都放在spath中
"""


def ModelTrainAndTest(trainedpd: pd.DataFrame, testpd: pd.DataFrame, spath: str,
                      modelpath: str = "Classifiers/saved_model/tmp", trainAgain: bool = True,
                      selectedFeature: List[str] = None):
    # 先生成模型 得到生成模型的准确率
    if not os.path.exists(spath):
        os.makedirs(spath)
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)
    if trainAgain:
        TrainThree(trainedpd, spath, modelpath, selectedFeature=selectedFeature)

    print("模型训练完成".center(40, "*"))
    print("开始对测试数据进行预测".center(40, "*"))
    testThree(testpd, spath, modelpath)
