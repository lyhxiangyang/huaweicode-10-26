import re
import time
from typing import List, Union, Set

import pandas as pd

from l3l2utils.DataFrameOperation import mergeDataFrames
from l3l2utils.DefineData import TIME_COLUMN_NAME, FAULT_FLAG, PID_FEATURE

"""
根据字符串自动得到时间串格式
"""


def getTimeFormat(onetime: str) -> str:
    T = ["%Y", "%m", "%d", "%H", "%M", "%S"]
    ilastpos = 0
    timeformatstr = ""
    for i, ipos in enumerate(re.finditer("\d+", onetime)):
        bpos = ipos.start()
        epos = ipos.end()
        timeformatstr += onetime[ilastpos:bpos]
        timeformatstr += T[i]
        ilastpos = epos
    return timeformatstr


"""
讲一个时间列表转化为标准格式 "%Y-%m-%d %H:%M:00" 
"""


def TranslateTimeListStrToStr(stime: List[str], leastTime: str = "%M") -> Union[
    str, list[str]]:
    # T = ["%Y", "%m", "%d", "%H", "%M", "%S"]
    changetoformat = '%Y-%m-%d %H:%M:%S'
    if leastTime == "%Y":
        changetoformat = "%Y-00-00 00:00:00"
    elif leastTime == "%m":
        changetoformat = "%Y-%m-00 00:00:00"
    elif leastTime == "%d":
        changetoformat = "%Y-%m-%d 00:00:00"
    elif leastTime == "%H":
        changetoformat = "%Y-%m-%d %H:00:00"
    elif leastTime == "%M":
        changetoformat = "%Y-%m-%d %H:%M:00"
    reslist = []
    for itime in stime:
        timeformat = getTimeFormat(itime)
        ttime = time.strptime(itime, timeformat)
        strtime = time.strftime(changetoformat, ttime)
        reslist.append(strtime)
    if len(reslist) == 1:
        return reslist[0]
    return reslist


# 将一个pd中的时间序列的秒变为00
def changeTimeFromOnepd(df: pd.DataFrame, leastTime: str = "%M",
                        timefeaturename: str = TIME_COLUMN_NAME) -> pd.DataFrame:
    if len(df) == 0:
        return df
    # timeformat给出的选项仅仅供选择，下面进行自动生成格式选项
    tpd = df.loc[:, [timefeaturename]].apply(
        lambda x: TranslateTimeListStrToStr(x.to_list(), leastTime=leastTime), axis=0)
    df.loc[:, timefeaturename] = tpd.loc[:, timefeaturename]
    return df


"""
将一个列表中的DataFrame的时间进行改变 变成%Y-%m-%d %H:%M:%S这种格式的
leastTime表示要精确到的位数，如%M  则代表这  %M以下的数据%S就是00 
"""


def changeTimeToFromPdlists(pds: List[pd.DataFrame], leastTime: str = "%M",
                            timefeaturename: str = TIME_COLUMN_NAME) -> List[pd.DataFrame]:
    changed_pds = []
    for ipd in pds:
        tpd = changeTimeFromOnepd(ipd, leastTime=leastTime, timefeaturename=timefeaturename)
        changed_pds.append(tpd)
    return changed_pds




'''
# - 功能介绍
# 将dataFrame中的一个名字为lable的列名字移动到最前面
# - 参数介绍
# 1. dataFrame是要移动的表格信息
# 2. 要修改的标签，如果没有这个标签，就不移动
# - 返回值介绍
# 放回第一个参数这个表格
'''


def pushLabelToFirst(dataFrame: pd.DataFrame, label: str) -> pd.DataFrame:
    columnsList = list(dataFrame.columns)
    if label not in columnsList:
        return dataFrame
    columnsList.insert(0, columnsList.pop(columnsList.index(label)))
    dataFrame = dataFrame[columnsList]
    return dataFrame


'''
# - 功能介绍
# 将dataFrame中的一个名字为lable的列名字移动到最后面
# - 参数介绍
# 1. dataFrame是要移动的表格信息
# 2. 要修改的标签，如果没有这个标签，就不移动
# - 返回值介绍
# 放回第一个参数这个表格
'''


def pushLabelToEnd(dataFrame: pd.DataFrame, label: str) -> pd.DataFrame:
    columnsList = list(dataFrame.columns)
    if label not in columnsList:
        return dataFrame
    columnsList.append(columnsList.pop(columnsList.index(label)))
    dataFrame = dataFrame[columnsList]
    return dataFrame


'''
-  功能介绍：
   用来一个DataFrame的列按照列名重新排序，使其列按照一定顺序排列
-  参数介绍：
   1. dataFrame是我们要排序的表
   2. reverse=False表示列名是按照字符串从小到大排列，True表示从大到小
-  返回值介绍：
   1. 表示排序好得到的DataFrame
'''


def sortLabels(dataFrame: pd.DataFrame, reverse=False) -> pd.DataFrame:
    columnsList = list(dataFrame.columns)
    columnsList.sort(reverse=reverse)
    dataFrame = dataFrame[columnsList]
    return dataFrame


# time  faultFlag  preFlag  mem_leak  mem_bandwidth
# 去除指定异常的首尾, 只去除首尾部分
def removeHeadTail_specifiedAbnormal(predictPd: pd.DataFrame, abnormals: Set[int], windowsize: int = 3) -> pd.DataFrame:
    dealflag = FAULT_FLAG
    def judge(x: pd.Series):
        # abnormals中有一个
        if len(abnormals & set(x)) != 0 and x.nunique() != 1:
            return False  # 表示去除
        else:
            return True  # 表示保留

    savelines = predictPd[dealflag].rolling(window=windowsize, min_periods=1).agg([judge])["judge"].astype("bool")
    return predictPd[savelines]

# 去除每个异常的首尾
def removeAllHeadTail(predictPd: pd.DataFrame, windowsize: int = 3, realFlagName: str = FAULT_FLAG) -> pd.DataFrame:
    allabnormals = set(predictPd[realFlagName])
    if 0 in allabnormals:
        allabnormals.remove(0)
    removepd = removeHeadTail_specifiedAbnormal(predictPd, windowsize=windowsize, abnormals=allabnormals)
    return removepd


# 去除进程数据中所有异常的首尾
# 保证这个进程数据包含pid选项
def removeProcessAllHeadTail(processPd: pd.DataFrame, windowsize: int = 3) -> pd.DataFrame:
    removepds = []
    for ipid, ipd in processPd.groupby(PID_FEATURE):
        if len(ipd) <= 6:
            continue
        tpd = removeAllHeadTail(ipd, windowsize=windowsize)
        removepds.append(tpd)
    allpd = mergeDataFrames(removepds)
    return allpd


# 去除指定异常及其首尾数据
def remove_Abnormal_Head_Tail(predictPd: pd.DataFrame, abnormals: Set[int], windowsize: int = 3) -> pd.DataFrame:
    dealflag = "faultFlag"

    def judge(x: pd.Series):
        # abnormals中有一个
        if len(abnormals & set(x)) != 0:
            return False  # 表示去除
        else:
            return True  # 表示保留

    savelines = predictPd[dealflag].rolling(window=windowsize, min_periods=1).agg([judge])["judge"].astype("bool")
    return predictPd[savelines]
