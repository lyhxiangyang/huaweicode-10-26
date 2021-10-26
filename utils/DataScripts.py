import os
import time
from typing import List, Union, Dict, Tuple, Any

import pandas as pd

from utils.DataFrameOperation import PushLabelToEnd, PushLabelToFirst, SortLabels, subtractLastLineFromDataFrame
from utils.DefineData import TIME_COLUMN_NAME, FAULT_FLAG, TIME_INTERVAL, CPU_FEATURE
from utils.FileSaveRead import saveDFListToFiles, saveCoreDFToFiles, saveFaultyDict

"""
将时间格式转化为int
"2021/8/29 0:54:08"
"""


def TranslateTimeToInt(stime: str, timeformat: str = '%Y-%m-%d %H:%M:%S') -> int:
    itime = time.mktime(time.strptime(stime, timeformat))
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


"""
将两个 int-DataFrame 合并在一起
合并的两个类型是fault-DataFrame
"""


def mergeTwoDF(dic1: Dict[int, pd.DataFrame], dic2: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
    allfaulty = list(dic1.keys())
    allfaulty.extend(list(dic2.keys()))
    allfauly = list(set(allfaulty))
    resDict = {}
    for ifaulty in allfauly:
        tpd: pd.DataFrame = pd.DataFrame()
        if ifaulty in dic1 and ifaulty in dic2:
            tpd = pd.concat([dic1[ifaulty], dic2[ifaulty]], ignore_index=True)
        elif ifaulty in dic1:
            tpd = dic1[ifaulty]
        elif ifaulty in dic2:
            tpd = dic2[ifaulty]
        resDict[ifaulty] = tpd
    return resDict


"""
将一个文件的所有时间进行连续时间段放入划分
index必须是0开头的
"""


def splitDataFrameByTime(df: pd.DataFrame, time_interval: int = 60, ) -> List[pd.DataFrame]:
    respd = []
    beginLine = 0
    sbeginLineTime = df.loc[beginLine, TIME_COLUMN_NAME]
    ibeginTime = TranslateTimeToInt(sbeginLineTime, '%Y-%m-%d %H:%M:%S')
    iLastLineTime = ibeginTime
    for nowline in range(1, len(df)):
        snowLineTime = df.loc[nowline, TIME_COLUMN_NAME]
        inowLineTime = TranslateTimeToInt(snowLineTime, '%Y-%m-%d %H:%M:%S')
        if inowLineTime - iLastLineTime == 0:
            continue
        # 误差在59 - 61s之间 或者等于0
        if not (time_interval - 1 <= inowLineTime - iLastLineTime <= time_interval + 1):
            tpd = df.loc[beginLine: nowline, :].reset_index(drop=True)
            beginLine = nowline
            respd.append(tpd)

        iLastLineTime = inowLineTime
    tpd = df.loc[beginLine: len(df), :].reset_index(drop=True)
    respd.append(tpd)
    return respd


"""
将数据按照核心来进行划分
"""


def SplitDFByCores(df: pd.DataFrame) -> List[Tuple[int, pd.DataFrame]]:
    if CPU_FEATURE not in df.columns.array:
        print("函数SplitCores错误")
        print("{} 这一列在表格中不存在".format(CPU_FEATURE))
        exit(1)
    corelist = list(set(df[CPU_FEATURE]))
    coreList = []
    for icore in corelist:
        tpd = df.loc[df[CPU_FEATURE] == icore]
        tpd.reset_index(drop=True, inplace=True)
        # 将CPU_FEATURE去掉
        # coreDict[icore] = tpd.drop(CPU_FEATURE, axis=1)
        coreList.append((icore, tpd))
    return coreList

"""
提取一个文件中的所有错误
"""
def abstractFaultPDDict(df: pd.DataFrame, extraFeature: List[str] = []) -> \
        Union[dict[int, dict], Any]:
    # 获得这个df中所有的错误码的类型
    if FAULT_FLAG not in df.columns.array:
        print("featureExtractionOriginalData 中没有错误标签")
        exit(1)
    # 获得所有的错误码标识
    faults = list(set(list(df.loc[:, FAULT_FLAG])))
    resFaultDF = {}
    for ifault in faults:
        selectLine = df.loc[:]
        fdf = df.loc[df.loc[:, FAULT_FLAG] == ifault, extraFeature]
        resFaultDF[ifault] = fdf
    return resFaultDF

"""
将一个process文件处理的过程
主要目的是获得 time-core-pd以及time-core-pd-faulty
"""

def processOneProcessFile(spath: str, filepd: pd.DataFrame, accumulationFeatures: List[str], process_features : List[str]):
    if not os.path.exists(spath):
        os.makedirs(spath)

    # 先按照时间段划分
    pdbytime = splitDataFrameByTime(filepd)

    # 将其保存到 spath/1.时间段划分集合
    print("1. 按照时间段划分开始")
    # tmp/{filename}/1.时间段划分集合
    saveDFListToFiles(spath=os.path.join(spath, "1.时间段划分集合文件"), pds=pdbytime)
    print("按照时间段划分结束")

    # 对每一个时间段划分
    thisFileFaulty_PD_Dict = {}
    thisTime_core_FileFaulty_PD_Dict = {}
    thisTime_core_PD_Dict = {}
    for i in range(0, len(pdbytime)):
        thisTime_core_PD_Dict[i] = {}
        thisTime_core_FileFaulty_PD_Dict[i] = {}

        print("2.{} 第{}个时间段依照核心划分".format(i, i))
        corepds = SplitDFByCores(pdbytime[i])
        # 将corepds保存出来 以便观察
        # tmp/tData/2.时间段划分集合文件详细信息/第{}时间段分割核心
        tcoresavepath = os.path.join(spath, "2.时间段划分集合文件详细信息", "{}.第{}时间段分割核心".format(i, i))
        saveCoreDFToFiles(tcoresavepath, corepds)
        # 对每个核心特征进行减去前一行
        subcorepds = []
        for icore, ipd in corepds:
            tpd = subtractLastLineFromDataFrame(ipd, accumulationFeatures)
            # 添加一个新的元素 cpu
            tpd['cpu'] = tpd['user'] + tpd['system']
            tpd = PushLabelToEnd(tpd, FAULT_FLAG)
            subcorepds.append((icore, tpd))

        # tmp/{filename}/2.时间段划分集合文件详细信息/
        tcoresavepath = os.path.join(spath, "2.时间段划分集合文件详细信息", "{}.第{}时间段分割核心-减前一行".format(i, i).format(i))
        saveCoreDFToFiles(tcoresavepath, subcorepds)

        # 对每一个核心进行处理
        tcoresavepath = os.path.join(spath, "2.时间段划分集合文件详细信息", "{}.第{}时间段分割核心-减去前一行-分割错误码".format(i, i))
        # 这个文件中错误：DF的字典结构
        for icore, icorepd in subcorepds:
            if icore not in thisTime_core_PD_Dict[i]:
                thisTime_core_PD_Dict[i][icore] = icorepd
            print("3.第{}时间段-{}核心处理中".format(i, icore))
            # 将所有的错误码进行提取
            FaultPDDict = abstractFaultPDDict(icorepd, extraFeature=process_features)
            if icore not in thisTime_core_FileFaulty_PD_Dict:
                thisTime_core_FileFaulty_PD_Dict[i][icore] = FaultPDDict
            #  将每个文件中的每个时间段中的每个核心进行错误码划分的结果进行保存
            tcore_fault_savepath = os.path.join(tcoresavepath, str(icore))
            saveFaultyDict(tcore_fault_savepath, FaultPDDict)
            # 合并总的错误
            thisFileFaulty_PD_Dict = mergeTwoDF(thisFileFaulty_PD_Dict, FaultPDDict)
    # 将这个文件中提取到的所有错误码进行保存
    tallsavefaultypath = os.path.join(spath, "3.所有错误码信息")
    saveFaultyDict(tallsavefaultypath, thisFileFaulty_PD_Dict)
    # 返回一个此文件所有错误的的Fault-PD， 返回按照时间段-核心-PD的字典结构， 返回按照时间段-核心-错误码-PD的字典结构
    return thisFileFaulty_PD_Dict, thisTime_core_PD_Dict, thisTime_core_FileFaulty_PD_Dict
