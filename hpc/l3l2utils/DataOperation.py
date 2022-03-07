import re
import time
from typing import List, Union, Set, Dict, Tuple, Any

import pandas as pd

from hpc.l3l2utils.DataFrameOperation import mergeDataFrames
from hpc.l3l2utils.DefineData import TIME_COLUMN_NAME, FAULT_FLAG, PID_FEATURE

"""
字符串转化为数字  按照分钟，秒
"""
def str_data_to_num(str_data)->int:
    # 格式时间成毫秒
    strptime = time.strptime(str_data,"%Y-%m-%d %H:%M:%S")
    mktime = int(time.mktime(strptime))
    return mktime
"""
将数字转换为 %Y-%m-%d %H:%M:%S 格式
"""
def num_to_str_data(num_data, timeformat: str = "%Y-%m-%d %H:%M:%S"):
    # 格式毫秒成指定格式时间
    num_data = time.localtime(num_data)  # 生成一个元祖式的时间
    strptime = time.strftime(timeformat,num_data) #格式化元祖
    return strptime

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
    str, List[str]]:
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
isremoveDuplicate true表示去除重复， false 表示不去除重复
"""


def changeTimeToFromPdlists(pds: List[pd.DataFrame], leastTime: str = "%M",
                            timefeaturename: str = TIME_COLUMN_NAME, isremoveDuplicate: bool = False) -> List[
    pd.DataFrame]:
    changed_pds = []
    print("一个pdlist改变时间".center(20, "*"))
    for i, ipd in enumerate(pds):
        tpd = changeTimeFromOnepd(ipd, leastTime=leastTime, timefeaturename=timefeaturename)
        if isremoveDuplicate:  # 将时间进行去重
            beforelen = len(tpd)
            tpd = tpd[~tpd[timefeaturename].duplicated()]
            tpd.reset_index(drop=True, inplace=True)
            afterlen = len(tpd)
            print("第{}个pd减少了{}行".format(i, beforelen - afterlen))
        changed_pds.append(tpd)
    return changed_pds

"""
将时间变成从数字变成字符串
"""
def changeTimePdsToStrFromInt(pds: List[pd.DataFrame]) -> List[pd.DataFrame]:
    for ipd in pds:
        ipd[TIME_COLUMN_NAME] = ipd[TIME_COLUMN_NAME].apply(lambda x: num_to_str_data(x))
    return pds


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
# faultFlag 针对[0,11,12] 这种  每个时间点的错误只能是一个
def removeHeadTail_specifiedAbnormal(predictPd: pd.DataFrame, abnormals: Set[List],
                                     windowsize: int = 3) -> pd.DataFrame:
    def judge(x: pd.Series):
        # x里面的种类有多种， 且和错误有交集
        if len(abnormals & set(x)) != 0 and x.nunique() != 1:
            return False  # 表示去除
        else:
            return True  # 表示保留

    savelines = predictPd[FAULT_FLAG].rolling(window=windowsize, min_periods=1).agg([judge])["judge"].astype("bool")
    return predictPd[savelines]


# 去除每个异常的首尾
def removeAllHeadTail(predictPd: pd.DataFrame, windowsize: int = 3) -> pd.DataFrame:
    allabnormals = set(predictPd[FAULT_FLAG])
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
    def judge(x: pd.Series):
        # abnormals中有一个
        if len(abnormals & set(x)) != 0:
            return False  # 表示去除
        else:
            return True  # 表示保留

    savelines = predictPd[FAULT_FLAG].rolling(window=windowsize, min_periods=1).agg([judge])["judge"].astype("bool")
    return predictPd[savelines]


# 去除每个异常及其首尾数据
def remove_AllAbnormalAndHeadTail(predictPd: pd.DataFrame, windowsize: int = 3) -> pd.DataFrame:
    allabnormals = set(predictPd[FAULT_FLAG])
    if 0 in allabnormals:
        allabnormals.remove(0)
    predictPd = remove_Abnormal_Head_Tail(predictPd, allabnormals, windowsize=windowsize)
    return predictPd


"""
对server数据列表进行改名字
返回一个新的列表
"""


def renamePds(datapds: List[pd.DataFrame], namedict: Dict):
    if len(namedict) == 0:
        return datapds
    renamepds = []
    for ipd in datapds:
        tpd = ipd.rename(columns=namedict, inplace=False)
        renamepds.append(tpd)
    return renamepds


"""
对process数据的每一个进程进行首尾的截取，去除没用的数据
"""
def removeProcessUselessData(datapds: List[pd.DataFrame], isremoveHead: bool=True, isremoveTail: bool=True, removelen:int=0) -> List[pd.DataFrame]:
    if not isremoveHead and not isremoveTail:
        return datapds
    if removelen == 0:
        return datapds

    processpds = []
    for iprocesspd in datapds:
        sumpdLists = []
        for ipid, ipd in iprocesspd.groupby(PID_FEATURE):
            # 先将一些不可用的数据进行清除,比如一个进程只运行了两分钟
            if len(ipd) <= 6:
                continue

            if isremoveHead and isremoveTail:
                ipd = ipd.iloc[removelen:-removelen, :]
            elif isremoveHead:
                ipd = ipd.iloc[removelen:, :]
            elif isremoveTail:
                ipd = ipd.iloc[:-removelen, :]
            sumpdLists.append(ipd)
        allsubtractpd = mergeDataFrames(sumpdLists)
        processpds.append(allsubtractpd)
    return processpds

"""
得到相同时间的pd
"""
def getsametimepd(servertimepd: pd.DataFrame, alltopdownspd: pd.DataFrame) -> Tuple[Any, Any]:
    def getSameTime(servertimes: List[str], topdowntimes: List[str]) -> List[str]:
        sametimes = sorted(list(set(servertimes) & set(topdowntimes)))
        return sametimes
    sametimes = getSameTime(servertimepd[TIME_COLUMN_NAME].tolist(), alltopdownspd[TIME_COLUMN_NAME].tolist())
    serverchooseindex = servertimepd[TIME_COLUMN_NAME].apply(lambda x: x in sametimes)
    return servertimepd[serverchooseindex].reset_index(drop=True)



def getRunHPCTimepdsFromProcess(pds: List[pd.DataFrame], processpds: List[pd.DataFrame])->List[pd.DataFrame]:
    def getSameTime(servertimes: List[str], topdowntimes: List[str]) -> List[str]:
        sametimes = sorted(list(set(servertimes) & set(topdowntimes)))
        return sametimes
    allprocesspd = mergeDataFrames(processpds)
    respdList = []
    for ipd in pds:
        sametimes = getSameTime(ipd[TIME_COLUMN_NAME].tolist(), allprocesspd[TIME_COLUMN_NAME].tolist())
        chooseindex = ipd[TIME_COLUMN_NAME].apply(lambda x: x in sametimes)
        npd = ipd[chooseindex].reset_index(drop=True)
        respdList.append(npd)
    return respdList








