import time
from typing import List, Union

import pandas as pd

from utils.DataFrameOperation import PushLabelToEnd, PushLabelToFirst, SortLabels
from utils.DefineData import TIME_COLUMN_NAME, FAULT_FLAG

"""
将时间格式转化为int
"2021/8/29 0:54:08"
"""


def TranslateTimeToInt(stime: str) -> int:
    itime = time.mktime(time.strptime(stime, '%Y-%m-%d %H:%M:%S'))
    return int(itime)


"""
将一个时间的中的秒字段都变为0
"""


def TranslateTimeStrToStr(stime: str, timeformat: str = '%Y-%m-%d %H:%M:%S') -> str:
    stime = stime[0]
    ttime = time.strptime(stime, timeformat)
    strtime = time.strftime('%Y-%m-%d %H:%M:00', ttime)
    return strtime


# 转变一个列表的字符串
def TranslateTimeListStrToStr(stime: List[str], timeformat: str = '%Y-%m-%d %H:%M:%S') -> Union[str, list[str]]:
    reslist = []
    for itime in stime:
        ttime = time.strptime(itime, timeformat)
        strtime = time.strftime('%Y-%m-%d %H:%M:00', ttime)
        reslist.append(strtime)
    if len(reslist) == 1:
        return reslist[0]
    return reslist


"""
只返回标准化之后的数据特征， 没有标准化的不返回
# 保留time和label
"""


def standardPDfromOriginal(df: pd.DataFrame, standardFeatures=None, meanValue=None) -> pd.DataFrame:
    if standardFeatures is None:
        standardFeatures = []
    nostandardDf = df.loc[:, standardFeatures]
    nostandardDf: pd.DataFrame
    # 如果为空 代表使用自己的mean
    if meanValue is None:
        meanValue = nostandardDf.mean()
    # 进行标准化
    standardDf = (nostandardDf / meanValue * 100).astype("int64")
    if TIME_COLUMN_NAME in df.columns.array:
        standardDf[TIME_COLUMN_NAME] = df[TIME_COLUMN_NAME]
    if FAULT_FLAG in df.columns.array:
        standardDf[FAULT_FLAG] = df[FAULT_FLAG]

    standardDf = SortLabels(standardDf)
    standardDf = PushLabelToFirst(standardDf, TIME_COLUMN_NAME)
    standardDf = PushLabelToEnd(standardDf, FAULT_FLAG)
    return standardDf
