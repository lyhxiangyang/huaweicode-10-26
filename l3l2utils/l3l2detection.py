import pandas as pd

"""
修改preFlag那些单独存在的点
"""


def fixIsolatedPoint(l2l3predetectresultpd: pd.DataFrame):
    def isAbnormal(x) -> bool: # 判断是否是
        if x == [0]:
            return False
        return True
    def isallEqual(abnormals, equalFlags): # 比较两个列表是否相等  abnormals是preFlag的截取， equalFlags 是
        if len(abnormals) != len(equalFlags):
            return False
        abnormals = [ 1 if isAbnormal(i) else 0 for i in abnormals] # 1代表异常 0 代表正常
        compareFlags = [abnormals[i] == equalFlags[i] for i in range(0, len(abnormals))] # True代表相同 False代表不同
        return compareFlags.count(True) == len(compareFlags)


    preflagList = list(l2l3predetectresultpd["preFlag"])
    for i in range(0, len(preflagList)):
        # 去除00100中的异常
        if 2<=i<=len(preflagList)-2 and isallEqual(preflagList[i-2:i+2], list(map(int, list("00100")))):
            preflagList[i] = [0]
            continue
        # 去除11011这种异常
        if 2<=i<=len(preflagList)-2 and isallEqual(preflagList[i-2:i+2], list(map(int, list("11011")))):
            preflagList[i] = sorted(list(set(preflagList[i - 1] + preflagList[i + 1])))
            continue

"""
对faultFlag进行修改
主要是将133变成131  134变成1
"""

def fixFaultFlag(l2l3predetectresultpd: pd.DataFrame):
    pass




