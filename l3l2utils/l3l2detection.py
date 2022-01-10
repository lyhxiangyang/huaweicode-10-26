from typing import List, Tuple

import pandas as pd

from l3l2utils.DataOperation import remove_Abnormal_Head_Tail
from l3l2utils.DefineData import FAULT_FLAG

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


def getTimePeriodInfo(l2l3predetectresultpd: pd.DataFrame, preflagname: str = "preFlag",
                      probabilityName: str = "probability") -> pd.DataFrame:
    # ========================================================================================================== 函数部分
    # 找到一个列表中连续不为0的数据的位置, 返回的是每段不为0的起始位置[4,9), 左闭右开
    def findAbnormalPos(flags: List[int]) -> List[Tuple[int, int]]:
        beginpos_endpos_List = []
        i = 0
        while i < len(flags):
            if flags[i] == 0:
                i += 1
                continue
            beginpos = i
            while i < len(flags) and flags[i] != 0:
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

    # 判断两个DataFrame是否交叉，如果交叉返回True，DataFrame  否则 False，DataFrame
    def determineTwoDataframeOverlap(df1: pd.DataFrame, df2: pd.DataFrame) -> Union[
        tuple[bool, None], tuple[bool, Any]]:
        df1times = set(df1[timeLabelName])
        df2times = set(df2[timeLabelName])
        overlapTime = list(df1times & df2times)
        if len(overlapTime) == 0:
            return False, None

        # 获得时间
        def f1(x):
            if x in overlapTime:
                return True
            return False

        return True, df1.loc[df1[timeLabelName].apply(f1)]

    # 判断一个DataFrame的时间是否与一个时间列表交叉，如果交叉返回交叉的True, DataFrame 否则 False，DataFrame
    # 返回是否交叉  返回交叉的部分  返回匹配到交叉的部分
    def determineDataframeListOverlap(df: pd.DataFrame, dflist: List[pd.DataFrame]) -> Union[
        tuple[bool, Any, Any], tuple[bool, None, None]]:
        for idf in dflist:
            iscross, crossdf = determineTwoDataframeOverlap(df, idf)
            if iscross:
                return True, crossdf, idf
        return False, None, None

    # 得到列表中出现的最大频率的数值，以及去重之后的列表
    def getMaxNumLabels(labels: List):
        prelabels = max(labels, key=labels.count)
        alllabeslList = sorted(list(set(labels)))
        return prelabels, alllabeslList

    """
    得到概率
    参数： iprepd-预测到的时间段  tcrosspd-交叉的时间段 trealpd-实际的时间段
    返回值 0: CPU概率   1: 内存泄露概率   2: 内存带宽异常

    """

    def getProbability(iprepd: pd.DataFrame, tcrosspd: pd.DataFrame, trealpd: pd.DataFrame) -> List:
        nowtimeProbability = iprepd[probabilityName].mean()
        return nowtimeProbability

    # ====================================================================================================== 函数部分结束
    # =================================================================================================得到时间段的逻辑部分
    testLabelName = testFlagName
    reallabels = list(predictpd[realFlagName])
    prelabels = list(predictpd[testLabelName])
    # =================================================================================================得到真实标签的分类
    beginpos_endpos_list = findAbnormalPos(reallabels)
    realTimePeriodAbnormalPds = getDataFramesFromPos(predictpd, beginpos_endpos_list)
    # =================================================================================================得到预测标签的分类
    beginpos_endpos_list = findAbnormalPos(prelabels)
    preTimePeriodAbnormalPds = getDataFramesFromPos(predictpd, beginpos_endpos_list)
    # =================================================================================================时间段的逻辑

    timeperiodDict = defaultdict(list)
    for iprepd in preTimePeriodAbnormalPds:
        assert len(iprepd) != 0
        prebegintime = iprepd[timeLabelName].iloc[0]  # 预测开始时间
        timeperiodDict["检测开始时间"].append(prebegintime)
        preendtime = iprepd[timeLabelName].iloc[-1]  # 预测结束时间
        timeperiodDict["检测结束时间"].append(preendtime)
        preLastime = len(iprepd)  # 预测持续时间
        timeperiodDict["检测运行时间"].append(preLastime)
        maxNumLabels, preAllLabels = getMaxNumLabels(list(iprepd[testLabelName]))  # 得到当前预测时间内的预测值
        timeperiodDict["检测标记"].append(maxNumLabels)
        timeperiodDict["检测所有标记"].append(",".join([str(i) for i in preAllLabels]))

        # 判断是否有真实标签值与其重叠
        iscross, tcrosspd, trealpd = determineDataframeListOverlap(iprepd, realTimePeriodAbnormalPds)
        # 得到概率
        nowtimeProbability = getProbability(iprepd=iprepd, tcrosspd=tcrosspd, trealpd=trealpd)
        timeperiodDict["概率"].append(nowtimeProbability)

        realcrossBeginTime = str(-1)
        realcrossEndTime = str(-1)
        crossTime = 0
        realTimeLen = 0
        realcrossLabels = 0
        if iscross:
            assert len(tcrosspd) != 0
            realcrossBeginTime = trealpd[timeLabelName].iloc[0]
            realcrossEndTime = trealpd[timeLabelName].iloc[-1]
            crossTime = len(tcrosspd)
            realTimeLen = len(trealpd)

            realcrossLabels, _ = getMaxNumLabels(list(trealpd[realFlagName]))
        timeperiodDict["实际开始时间"].append(realcrossBeginTime)
        timeperiodDict["实际结束时间"].append(realcrossEndTime)
        timeperiodDict["实际运行时间"].append(realTimeLen)
        timeperiodDict["重叠时间"].append(crossTime)
        timeperiodDict["实际标记"].append(realcrossLabels)

    timeperiodDictPd = pd.DataFrame(data=timeperiodDict)
    return timeperiodDictPd
