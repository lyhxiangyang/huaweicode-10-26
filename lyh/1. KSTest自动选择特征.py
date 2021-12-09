import os
from typing import List

import pandas as pd

from utils.DataFrameOperation import mergeDataFrames
from utils.DefineData import FAULT_FLAG, TIME_COLUMN_NAME
from utils.FeatureSelection import getPValueFromTwoDF, getFeatureNameByBenjamini_Yekutiel
from utils.auto_forecast import getfilespath, getfilepd, differenceServer, removeAllHeadTail

"""
比较两个server的DataFrame 异常为abnormal
"""
def getServerDiffFeaturesFromTwoData(normalserverPD: pd.DataFrame, abnormalserverPD: pd.DataFrame, abnormaltype: int) -> List[str]:
    # 将所有的异常的首尾去掉
    abnormalserverPD = removeAllHeadTail(abnormalserverPD, windowsize=3)
    abnormal_pdDict = dict(list(abnormalserverPD.groupby(FAULT_FLAG))) # 指定异常类型
    specialAbnormalPd  = abnormal_pdDict[abnormaltype]
    print("正常数据长度：{}".format(len(normalserverPD)))
    print("异常特征长度：{}".format(len(specialAbnormalPd)))
    # 得到各个特征对应的pvalue值
    feature_pvalueDict = getPValueFromTwoDF(normalserverPD, specialAbnormalPd)
    selectFeatureDict, notselectFeatureDict = getFeatureNameByBenjamini_Yekutiel(feature_pvalueDict, fdr=0.05)
    selectFeatureList = sorted(selectFeatureDict.items(), key=lambda item: item[1], reverse=False)
    notselectFeatureList = sorted(notselectFeatureDict.items(), key=lambda item: item[1], reverse=False)
    print("特征总个数:{}".format(len(selectFeatureList) + len(notselectFeatureList)))
    print("选择的特征".center(40, "*"))
    print("个数: {}".format(len(selectFeatureList)))
    print(selectFeatureList)
    print("没有选择的特征".center(40, "*"))
    print("个数：{}".format(len(notselectFeatureList)))
    print(notselectFeatureList)
    return selectFeatureList

def getServerDiffFeaturesFromOneData(abnormalserverPD: pd.DataFrame, abnormaltype: int) -> List[str]:
    abnormalserverPD = removeAllHeadTail(abnormalserverPD, windowsize=3)
    abnormal_pdDict = dict(list(abnormalserverPD.groupby(FAULT_FLAG))) # 指定异常类型
    specialAbnormalPd  = abnormal_pdDict[abnormaltype] # 异常类型
    normalserverPD = abnormalserverPD[0]
    print("正常数据长度：{}".format(len(normalserverPD)))
    print("异常特征长度：{}".format(len(specialAbnormalPd)))
    # =====================================================得到特征选择
    feature_pvalueDict = getPValueFromTwoDF(normalserverPD, specialAbnormalPd)
    selectFeatureDict, notselectFeatureDict = getFeatureNameByBenjamini_Yekutiel(feature_pvalueDict, fdr=0.05)
    selectFeatureList = sorted(selectFeatureDict.items(), key=lambda item: item[1], reverse=False)
    notselectFeatureList = sorted(notselectFeatureDict.items(), key=lambda item: item[1], reverse=False)
    print("特征总个数:{}".format(len(selectFeatureList) + len(notselectFeatureList)))
    print("选择的特征".center(40, "*"))
    print("个数: {}".format(len(selectFeatureList)))
    print(selectFeatureList)
    print("没有选择的特征".center(40, "*"))
    print("个数：{}".format(len(notselectFeatureList)))
    print(notselectFeatureList)
    return selectFeatureList


if __name__ == "__main__":
    predictdirpath = R"C:\Users\lWX1084330\Desktop\正常和异常数据\训练数据-E5-1km-异常数据"
    predictserverfiles = getfilespath(os.path.join(predictdirpath, "server"))
    predictprocessfiles = getfilespath(os.path.join(predictdirpath, "process"))

    normaldirpath = R"C:\Users\lWX1084330\Desktop\正常和异常数据\E5-3km-正常数据"
    normalserverfiles = getfilespath(os.path.join(normaldirpath, "server"))
    normalprocessfiles = getfilespath(os.path.join(normaldirpath, "process"))

    # 特征值
    server_feature = [
        # "time",
        "user",
        "nice",
        "system",
        "idle",
        "iowait",
        "irq",
        "softirq",
        "steal",
        "guest",
        "guest_nice",
        "ctx_switches",
        "interrupts",
        "soft_interrupts",
        "syscalls",
        "freq",
        "load1",
        "load5",
        "load15",
        "total",
        "available",
        "percent",
        "used",
        "free",
        "active",
        "inactive",
        "buffers",
        "cached",
        "handlesNum",
        "pgpgin",
        "pgpgout",
        "fault",
        "majflt",
        "pgscank",
        "pgsteal",
        "pgfree",
        # "faultFlag",
    ]
    server_accumulate_feature = ['idle', 'iowait', 'interrupts', 'user', 'system', 'ctx_switches', 'soft_interrupts', 'irq',
                  'softirq', 'steal', 'syscalls', 'handlesNum', 'pgpgin', 'pgpgout', 'fault', 'majflt', 'pgscank',
                  'pgsteal', 'pgfree']

    # ============================================================================================= 先将正常数据和预测数据的指标从磁盘中加载到内存中
    print("将数据从文件中读取".center(40, "*"))
    predictserverpds = []
    normalserverpds = []
    # 添加上时间和FAULT_FLAG
    time_server_feature = server_feature.copy()
    time_server_feature.extend([TIME_COLUMN_NAME, FAULT_FLAG])
    for ifile in predictserverfiles:
        tpd = getfilepd(ifile)
        tpd = tpd.loc[:, time_server_feature]
        predictserverpds.append(tpd)
    for ifile in normalserverfiles:
        tpd = getfilepd(ifile)
        tpd = tpd.loc[:, time_server_feature]
        normalserverpds.append(tpd)
    # ============================================================================================= 对读取到的数据进行差分，并且将cpu添加到要提取的特征中
    print("对读取到的原始数据进行差分".format(40, "*"))
    normalserverpds = differenceServer(normalserverpds, server_accumulate_feature)
    predictserverpds = differenceServer(predictserverpds, server_accumulate_feature)
    # ============================================================================================= 不进行标准化 直接合并判断
    allnormalserverpd, _ = mergeDataFrames(normalserverpds)
    allabnormalserverpd, _ = mergeDataFrames(predictserverpds)
    # getServerDiffFeaturesFromTwoData(allnormalserverpd, allabnormalserverpd, 55)

    print("将两组数据中的正常和正常进行对比: ".center(20, "#"))
    normal_normal_selectFeatureList = getServerDiffFeaturesFromTwoData(allnormalserverpd, allabnormalserverpd, 0)


    print("将一组数据中的正常和异常进行对比：".center(20, "#"))
    selectFeatureList = getServerDiffFeaturesFromOneData(allabnormalserverpd, 55)

    print("去掉正常和正常时选择的指标：".center(20, "#"))
    print("选择的指标：")
    print("{}".format(list(set(selectFeatureList) - set(normal_normal_selectFeatureList))))



















