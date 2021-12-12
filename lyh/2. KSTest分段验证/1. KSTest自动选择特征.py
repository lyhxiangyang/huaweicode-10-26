import os
from typing import List, Dict

import pandas as pd

from utils.DataFrameOperation import mergeDataFrames
from utils.DataScripts import standardPDfromOriginal1, removeTimeAndfaultFlagFromList
from utils.DefineData import FAULT_FLAG, TIME_COLUMN_NAME
from utils.FeatureSelection import getPValueFromTwoDF, getFeatureNameByBenjamini_Yekutiel
from utils.auto_forecast import getfilespath, getfilepd, differenceServer, removeAllHeadTail

"""
比较两个server的DataFrame 异常为abnormal
"""
def getServerDiffFeaturesFromTwoData(normalserverPD: pd.DataFrame, abnormalserverPD: pd.DataFrame, abnormaltype: int, spath: str = ".") -> Dict[str, float]:
    if not os.path.exists(spath):
        os.makedirs(spath)
    # 将所有的异常的首尾去掉
    abnormalserverPD = removeAllHeadTail(abnormalserverPD, windowsize=3)
    abnormal_pdDict = dict(list(abnormalserverPD.groupby(FAULT_FLAG))) # 指定异常类型
    specialAbnormalPd  = abnormal_pdDict[abnormaltype]
    print("正常数据长度：{}".format(len(normalserverPD)))
    print("异常{}特征长度：{}".format(abnormaltype, len(specialAbnormalPd)))

    # =============================================================================================================归一化
    columnsFeas = removeTimeAndfaultFlagFromList(list(normalserverPD.columns))
    # ==============正常数据的归一化
    print("正常数据平均值和最大值的比较".center(40, "*"))
    # print(normalserverPD[columnsFeas].rolling(window=len(normalserverPD[columnsFeas]), min_periods=len(normalserverPD[columnsFeas])).agg(["max", "mean"]).dropna())
    normalserverPD[columnsFeas].rolling(window=len(normalserverPD[columnsFeas]),
                                        min_periods=len(normalserverPD[columnsFeas])).agg(["max", "mean"]).dropna().to_csv(os.path.join(spath, "1. 正常数据平均值和最大值的比较.csv"))
    # 正常上数据使用最大值
    meanValue = normalserverPD[columnsFeas].mean()
    normalserverPD = standardPDfromOriginal1(df=normalserverPD, meanValue=meanValue, standardFeatures=columnsFeas, standardValue=1, standardValueType="float")
    # ===============异常数据归一化  使用异常里面的正常数据的最大值
    print("异常数据中正常值的平均值和最大值的比较".center(40, "*"))
    # print(abnormal_pdDict[0].rolling(window=len(abnormal_pdDict[0]), min_periods=len(abnormal_pdDict[0])).agg(["max", "mean"]).dropna())
    abnormal_pdDict[0].rolling(window=len(abnormal_pdDict[0]), min_periods=len(abnormal_pdDict[0])).agg(
        ["max", "mean"]).dropna().to_csv(os.path.join(spath, "2. 异常数据中正常值的平均值和最大值的比较.csv"))
    meanValue = abnormal_pdDict[0][columnsFeas].mean()
    specialAbnormalPd = standardPDfromOriginal1(df=specialAbnormalPd, meanValue=meanValue, standardFeatures=columnsFeas, standardValue=1, standardValueType="float")

    normalserverPD.to_csv(os.path.join(spath, "3. 归一化后正常数据中正常值.csv"))
    specialAbnormalPd.to_csv(os.path.join(spath, "4. 归一化后异常数据中{}异常值.csv".format(abnormaltype)))
    # =============================================================================================================归一化结束

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
    return selectFeatureDict

def getServerDiffFeaturesFromOneData(abnormalserverPD: pd.DataFrame, abnormaltype: int, spath: str = ".") -> Dict[str, float]:
    if not os.path.exists(spath):
        os.makedirs(spath)
    abnormalserverPD = removeAllHeadTail(abnormalserverPD, windowsize=3)
    abnormal_pdDict = dict(list(abnormalserverPD.groupby(FAULT_FLAG))) # 指定异常类型
    specialAbnormalPd  = abnormal_pdDict[abnormaltype] # 异常类型
    normalserverPD = abnormal_pdDict[0]
    print("正常数据长度：{}".format(len(normalserverPD)))
    print("异常特征{} 长度：{}".format(abnormaltype, len(specialAbnormalPd)))

    # =============================================================================================================归一化
    columnsFeas = removeTimeAndfaultFlagFromList(list(normalserverPD.columns))
    # ==============正常数据的归一化
    print("异常数据中正常平均值和最大值的比较".format(abnormaltype).center(40, "*"))
    # print(normalserverPD[columnsFeas].rolling(window=len(normalserverPD[columnsFeas]), min_periods=len(normalserverPD[columnsFeas])).agg(["max", "mean"]).dropna())
    normalserverPD[columnsFeas].rolling(window=len(normalserverPD[columnsFeas]),
                                        min_periods=len(normalserverPD[columnsFeas])).agg(["max", "mean"]).dropna().to_csv(os.path.join(spath, "1. 异常数据中正常值平均值和最大值的比较.csv"))
    # 正常上数据使用最大值
    meanValue = normalserverPD[columnsFeas].mean()
    normalserverPD = standardPDfromOriginal1(df=normalserverPD, meanValue=meanValue, standardFeatures=columnsFeas, standardValue=1, standardValueType="float")
    # ===============异常数据归一化  使用异常里面的正常数据的最大值
    print("异常数据中正常值的平均值和最大值的比较".center(40, "*"))
    # print(abnormal_pdDict[0].rolling(window=len(abnormal_pdDict[0]), min_periods=len(abnormal_pdDict[0])).agg(["max", "mean"]).dropna())
    abnormal_pdDict[0].rolling(window=len(abnormal_pdDict[0]), min_periods=len(abnormal_pdDict[0])).agg(
        ["max", "mean"]).dropna().to_csv(os.path.join(spath, "2. 异常数据中正常值平均值和最大值的比较.csv"))
    meanValue = abnormal_pdDict[0][columnsFeas].mean()
    specialAbnormalPd = standardPDfromOriginal1(df=specialAbnormalPd, meanValue=meanValue, standardFeatures=columnsFeas, standardValue=1, standardValueType="float")

    normalserverPD.to_csv(os.path.join(spath, "3. 归一化后正常数据中正常值.csv"))
    specialAbnormalPd.to_csv(os.path.join(spath, "4. 归一化后异常数据中{}异常值".format(abnormaltype)))
    # =============================================================================================================归一化结束



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
    return selectFeatureDict


if __name__ == "__main__":
    predictdirpath = R"C:\Users\lWX1084330\Desktop\正常和异常数据\训练数据-E5-1km-异常数据"
    predictserverfiles = getfilespath(os.path.join(predictdirpath, "server"))
    predictprocessfiles = getfilespath(os.path.join(predictdirpath, "process"))

    normaldirpath = R"C:\Users\lWX1084330\Desktop\正常和异常数据\E5-3km-正常数据"
    normalserverfiles = getfilespath(os.path.join(normaldirpath, "server"))
    normalprocessfiles = getfilespath(os.path.join(normaldirpath, "process"))

    abnormalType = 55
    spath = "tmp/特征比较"

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

    tpath = os.path.join(spath, "1.两组数据正常-正常")
    print("将两组数据中的正常和正常进行对比: ".center(80, "-"))
    normal_normal_selectFeatureDict = getServerDiffFeaturesFromTwoData(allnormalserverpd, allabnormalserverpd, 0, spath=tpath)

    tpath = os.path.join(spath, "2.一组数据正常和异常")
    print("将一组数据中的正常和异常进行对比：".center(80, "-"))
    selectFeatureDict = getServerDiffFeaturesFromOneData(allabnormalserverpd, abnormalType, spath=tpath)
    print("去掉正常和正常时选择的指标：")
    print("{}".format(list(set(selectFeatureDict.keys()) - set(normal_normal_selectFeatureDict.keys()))))

    tpath = os.path.join(spath, "3.两组数据正常和异常")
    print("将两组数组中正常和异常进行对比".center(80, "-"))
    normal_abnormal_selectFeatureDict = getServerDiffFeaturesFromTwoData(allnormalserverpd, allabnormalserverpd, abnormalType, spath=tpath)
    print("去掉正常和正常时选择的指标：")
    print("{}".format(list(set(normal_abnormal_selectFeatureDict.keys()) - set(normal_normal_selectFeatureDict.keys()))))




















