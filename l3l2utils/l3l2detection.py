from itertools import chain
from typing import List, Tuple, Dict

import pandas as pd

from l3l2utils.DataOperation import remove_Abnormal_Head_Tail
from l3l2utils.DefineData import FAULT_FLAG, TIME_COLUMN_NAME, errorFeatureDict

"""
修改preFlag那些单独存在的点
"""


def fixIsolatedPoint(l2l3predetectresultpd: pd.DataFrame):
    l2l3predetectresultpd = l2l3predetectresultpd.copy()

    def isAbnormal(x) -> bool:  # 判断是否是
        if x == [0]:
            return False
        return True

    def isallEqual(abnormals, equalFlags):  # 比较两个列表是否相等  abnormals是preFlag的截取， equalFlags 是
        if len(abnormals) != len(equalFlags):
            return False
        abnormals = [1 if isAbnormal(i) else 0 for i in abnormals]  # 1代表异常 0 代表正常
        compareFlags = [abnormals[i] == equalFlags[i] for i in range(0, len(abnormals))]  # True代表相同 False代表不同
        return compareFlags.count(True) == len(compareFlags)

    preflagList = list(l2l3predetectresultpd["preFlag"])
    for i in range(0, len(preflagList)):
        # 去除00100中的异常
        if 2 <= i <= len(preflagList) - 2 and isallEqual(preflagList[i - 2:i + 2], list(map(int, list("00100")))):
            preflagList[i] = [0]
            continue
        # 去除11011这种异常
        if 2 <= i <= len(preflagList) - 2 and isallEqual(preflagList[i - 2:i + 2], list(map(int, list("11011")))):
            preflagList[i] = sorted(list(set(preflagList[i - 1] + preflagList[i + 1])))
            continue
    l2l3predetectresultpd["preFlag"] = preflagList
    return l2l3predetectresultpd


"""
对faultFlag进行修改
主要是将133变成131  134变成132
99 状态代表着这段时间没有wrf运行，可以删除
此外还要删除41-45  71-75  91-95
"""


def fixFaultFlag(l2l3predetectresultpd: pd.DataFrame):
    l2l3predetectresultpd = l2l3predetectresultpd.copy()
    l2l3predetectresultpd = remove_Abnormal_Head_Tail(l2l3predetectresultpd,
                                                      abnormals={41, 42, 43, 44, 45, 71, 72, 73, 74, 75, 91, 92, 93, 94,
                                                                 95, 99}, windowsize=4)
    # 修改标签
    l2l3predetectresultpd.loc[:, FAULT_FLAG] = l2l3predetectresultpd.loc[:, FAULT_FLAG].apply(
        lambda x: 131 if x == 133 else x)
    l2l3predetectresultpd.loc[:, FAULT_FLAG] = l2l3predetectresultpd.loc[:, FAULT_FLAG].apply(
        lambda x: 132 if x == 134 else x)
    return l2l3predetectresultpd


"""
函数功能： 得到每个时间点的概率
"""


def getDetectionProbability(preFlagsList: List[List[int]]):
    def getBeforDataWeight(preFlags: List[List[int]], ipos: int, ifault: int, judgelen: int) -> List[float]:
        res = []
        for bpos in range(ipos - judgelen, ipos, 1):
            if bpos < 0:
                res.append(0)
                continue
            if ifault in preFlags[bpos]:
                res.append(1)
                continue
            res.append(0)
        return res

    def getAfterDataWeight(preFlags: List[List], ipos: int, ifault: int, judgelen: int) -> List[float]:
        res = []
        for bpos in range(ipos + 1, ipos + judgelen + 1, 1):
            if bpos >= len(preFlags):
                res.append(0)
                continue
            if ifault in preFlags[bpos]:
                res.append(1)
                continue
            res.append(0)
        return res

    def getp(weights: List[float], probabilities: List[float]) -> float:
        assert len(weights) == len(probabilities)
        return sum(weights[i] * probabilities[i] for i in range(0, len(weights)))

    # 得到时间概率的逻辑部分
    probabilityList = []
    preprobability = [0.1, 0.2, 0.3, 0.4]
    reverseprobability = list(reversed(preprobability))  # 概率相反
    judgeLen = len(preprobability)
    for ipos, onepreflag in enumerate(preFlagsList):
        oneprobabilityDict = {}
        for iflag in onepreflag:
            beforeWeight = getBeforDataWeight(preFlagsList, ipos, iflag, judgeLen)
            afterWeight = getAfterDataWeight(preFlagsList, ipos, iflag, judgeLen)
            nowprobability = 0.1 + 0.45 * (getp(beforeWeight, preprobability)) + 0.45 * (
                getp(afterWeight, reverseprobability))
            oneprobabilityDict[iflag] = nowprobability
        probabilityList.append(oneprobabilityDict)
    return probabilityList


"""
函数功能：根据预测信息得到一段时间内的每个时间段信息及其概率
"""


def getTimePeriodInfo(l2l3predetectresultpd: pd.DataFrame, preflagName: str = "preFlag",
                      probabilityName: str = "probability") -> pd.DataFrame:
    # ========================================================================================================== 函数部分
    # 找到一个列表中连续不为0的数据的位置, 返回的是每段不为0的起始位置[4,9), 左闭右开
    def findAbnormalPos(flags: List[int]) -> List[Tuple[int, int]]:
        beginpos_endpos_List = []
        i = 0
        flagslen = len(flags)
        while i < flagslen:
            if flags[i] == [0]:
                i += 1
                continue
            beginpos = i
            while i < flagslen and flags[i] != [0]:
                i += 1
            endpos = i
            beginpos_endpos_List.append((beginpos, endpos))
        return beginpos_endpos_List

    # 根据位置的起始位置得到DataFrame
    def getDataFramesFromPos(pd: pd.DataFrame, pos: List[Tuple[int, int]]) -> List[pd.DataFrame]:
        pd = pd.copy()
        respdList = []
        for i in pos:
            # loc和使用[]是不一样的
            respdList.append(pd[i[0]:i[1]])
        return respdList

    # 计算一段时间的概率 ifault的概率
    def getProbability(prepd: pd.DataFrame, ifault: int, probabilityName: str = "probability") -> float:
        faultprob = 0
        prelen = len(prepd)
        probabilityLists = prepd[probabilityName]
        for iprobability in probabilityLists:
            iprobability: Dict
            if ifault in iprobability:
                faultprob += iprobability[ifault] / prelen
        return faultprob

    """
    得到预测值的错误类型 返回List， 概率从大到小排序
    返回一个List
    """

    def getErrorInfo(prepd: pd.DataFrame, preflagName: str = "preFlag", probabilityName: str = "probability") -> List:
        errorList = []
        preFlags = prepd[preflagName]
        allPreFlags = sorted(list(set(chain.from_iterable(preFlags))))
        for ifault in allPreFlags:
            errorDict = {}
            errorDict["type"] = ifault
            # 计算概率
            errorDict["probability"] = getProbability(prepd, ifault, probabilityName)
            errorList.append(errorDict)
        # 需要对errorList按照概率从大到小进行排序
        errorList = sorted(errorList, key=lambda x: x["probability"], reverse=True)
        return errorList

    """
    得到预测值所依赖的特征值及其重要程度
    将预测的这段时间内每个错误对应的特征值使用的特征百分比作为参考依据
    返回一个Dict
    """
    def getKPIInfo(prepd: pd.DataFrame,  preflagName: str = "preFlag"):
        preFlags = prepd[preflagName]
        repeatPreFlags = list(chain.from_iterable(preFlags))
        alluserfeatureList = []
        for ifault in repeatPreFlags:
            if ifault not in errorFeatureDict:
                print("{} 的特征值没有存储".format(ifault))
                continue
            usedfeatures = errorFeatureDict[ifault]
            alluserfeatureList.extend(usedfeatures)

        # 得到所有错误使用的featurename，然后判断其占比进行输出
        alluserfeatureSeries = pd.Series(data=alluserfeatureList)
        proportionDict = dict(alluserfeatureSeries.value_counts(normalize=True, ascending=False))
        return proportionDict
    """
    返回影响程序的总体指标
    返回值float
    """
    def getperformance_var(prepd: pd.DataFrame, preflagName: str = "preFlag", probabilityName: str = "probability")->float:
        prepdlen = len(prepd)
        return -(1 - prepdlen / 10)

    if FAULT_FLAG in l2l3predetectresultpd.columns:
        reallabels = list(l2l3predetectresultpd[FAULT_FLAG])
        beginpos_endpos_list = findAbnormalPos(reallabels)
        realTimePeriodAbnormalPds = getDataFramesFromPos(l2l3predetectresultpd, beginpos_endpos_list)

    # 预测标签列表值
    prelabels = list(l2l3predetectresultpd[preflagName])
    beginpos_endpos_list = findAbnormalPos(prelabels)
    preTimePeriodAbnormalPds = getDataFramesFromPos(l2l3predetectresultpd, beginpos_endpos_list)
    # 用于json格式输出的Dict，包含时间段格式
    outputTimePeriodList = []
    for iprepd in preTimePeriodAbnormalPds:
        oneTimePeriodDict = {}
        oneTimePeriodDict["begin_time"] = iprepd[TIME_COLUMN_NAME].iloc[0]  # 预测开始时间
        oneTimePeriodDict["end_time"] = iprepd[TIME_COLUMN_NAME].iloc[-1]  # 预测结束时间
        oneTimePeriodDict["error_info"] = getErrorInfo(iprepd, preflagName, probabilityName)
        oneTimePeriodDict["kpi"] = getKPIInfo(iprepd, preflagName)
        oneTimePeriodDict["performance_var"] = getperformance_var(iprepd, preflagName, probabilityName)
        # 将这个时间段中的数据添加
        outputTimePeriodList.append(oneTimePeriodDict)
    return outputTimePeriodList
