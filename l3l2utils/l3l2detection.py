from typing import List

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
得到概率
得到每一个点内各个异常的概率
如果之前的
"""


def getDetectionProbability(l2l3predetectresultpd: pd.DataFrame):
    def getBeforData(preFlags: List[List[int]], ipos: int, ifault: int, judgelen: int)->List[int]:
        res = []
        for bpos in (ipos - judgelen, ipos, 1):
            if bpos < 0:
                res.append(0)
                continue
            if ifault in preFlags[ipos]:
                res.append(1)
            else:
                res.append(0)
        return res
    def getAfterData(preFlags: List[List], ipos: int, ifault: int, judgelen: int) -> List[int]:
        pass
    def getp(isSameFlags: List[int], probabilities: List[float]) -> float:
        assert len(isSameFlags) == len(probabilities)
        return sum(isSameFlags[i] * probabilities[i] for i in range(0, len(isSameFlags)))












