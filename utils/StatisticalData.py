"""
和统计有关系的文件
"""
from typing import List

import pandas as pd

from utils.auto_forecast import getDetailedInformationOnTime

"""
函数功能: 将检测的到的标签值去除不好的点
不会修改传入的参数
"""
def removeAbnormalPoints(df: pd.DataFrame, testLabelName: str = "preFlag") -> pd.DataFrame:
    winsize = 7
    df = df.copy()
    def getRightLabel(x: pd.Series):
        x = list(x)
        nowlable = x[len(x) - winsize // 2 - 1]
        maxlabel = max(x, key=x.count)
        if x.count(nowlable) == x.count(maxlabel):
            return nowlable
        return maxlabel
    df[testLabelName] = df[testLabelName].rolling(window=winsize, center=True, min_periods=1).agg([getRightLabel])
    return df

def makeProbabilityToFloat(probability: List[str]) -> List[float]:
    def trans(s: str)->float:
        return float(s[0:-1]) / 100
    return [trans(i) for i in probability]


def getTimePeriod(df: pd.DataFrame, timename: str = "time", realLabelName: str = "faultFlag", testLabelName: str = "preFlag", probabilityName="概率") -> pd.DataFrame:
    tpd = getDetailedInformationOnTime(df, timeLabelName=timename, realFlagName=realLabelName, testFlagName=testLabelName, probabilityName=probabilityName)
    feas = ["检测开始时间", "实际开始时间", "检测结束时间", "实际结束时间", "检测标记", "实际标记", "概率", "检测运行时间","实际运行时间","重叠时间"]
    return tpd[feas]



