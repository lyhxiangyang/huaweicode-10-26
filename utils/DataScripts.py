import os
import time
from collections import defaultdict
from typing import List, Union, Dict, Tuple, Any
import pandas as pd

from utils.DataFrameOperation import PushLabelToEnd, PushLabelToFirst, SortLabels, subtractLastLineFromDataFrame
from utils.DefineData import TIME_COLUMN_NAME, FAULT_FLAG, TIME_INTERVAL, CPU_FEATURE, MODEL_TYPE
from utils.FeatureExtraction import featureExtractionUsingFeatures
from utils.FileSaveRead import saveDFListToFiles, saveCoreDFToFiles, saveFaultyDict

"""
将时间格式转化为int
"2021/8/29 0:54:08"
"""


def TranslateTimeToInt(stime: str, timeformat: str = '%Y-%m-%d %H:%M:%S') -> int:
    itime = time.mktime(time.strptime(stime, timeformat))
    return int(itime)


"""
将时间格式转化为str
"2021/8/29 0:54:08"
"""


def TranslateTimeToStr(nowtime: int, timeformat: str = '%Y-%m-%d %H:%M:%S') -> str:
    struct_time = time.localtime(nowtime)
    return time.strftime(timeformat, struct_time)


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


def standardPDfromOriginal(df: pd.DataFrame, standardFeatures=None, meanValue=None,
                           standardValue: int = 100) -> pd.DataFrame:
    if standardFeatures is None:
        standardFeatures = []
    nostandardDf = df.loc[:, standardFeatures]
    nostandardDf: pd.DataFrame
    # 如果为空 代表使用自己的mean
    if meanValue is None:
        meanValue = nostandardDf.mean()
    # 进行标准化
    standardDf = (nostandardDf / meanValue * standardValue).astype("int64")
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


def splitDataFrameByTime(df: pd.DataFrame, time_interval: int = 60, timeformat: str = '%Y-%m-%d %H:%M:%S') -> List[
    pd.DataFrame]:
    respd = []
    beginLine = 0
    sbeginLineTime = df.loc[beginLine, TIME_COLUMN_NAME]
    ibeginTime = TranslateTimeToInt(sbeginLineTime, timeformat)
    iLastLineTime = ibeginTime
    for nowline in range(1, len(df)):
        snowLineTime = df.loc[nowline, TIME_COLUMN_NAME]
        inowLineTime = TranslateTimeToInt(snowLineTime, timeformat)
        if inowLineTime - iLastLineTime == 0:
            continue
        # 误差在59 - 61s之间 或者等于0
        if not (-time_interval - 1 <= inowLineTime - iLastLineTime <= time_interval + 1):
            tpd = df.loc[beginLine: nowline - 1, :].reset_index(drop=True)
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


def processOneProcessFile(spath: str, filepd: pd.DataFrame, accumulationFeatures: List[str],
                          process_features: List[str]):
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
        # 提取的指标多加一个'cpu'
        if 'cpu' not in process_features:
            process_features.insert(2, 'cpu')

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
            if icore not in thisTime_core_FileFaulty_PD_Dict[i]:
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


## ==== 用于process的步骤2

"""
得到平均值
"""


def getDFmean(df: pd.DataFrame, standardFeatures: List[str]) -> pd.Series:
    if FAULT_FLAG in standardFeatures:
        standardFeatures.remove(FAULT_FLAG)
    if TIME_COLUMN_NAME in standardFeatures:
        standardFeatures.remove(TIME_COLUMN_NAME)
    return df.loc[:, standardFeatures].mean()


"""
作用：标准化 file-time-core 
返回值： file-time-core的字典结构
"""


def standard_file_time_coreDict(ftcPD, standardFeature, meanvalue, standardValue: int):
    resDict = {}
    for filename, time_core_pdDict in ftcPD.items():
        resDict[filename] = {}
        for time, core_pdDict in time_core_pdDict.items():
            resDict[filename][time] = {}
            for icore, tpd in core_pdDict.items():
                resDict[filename][time][icore] = standardPDfromOriginal(tpd, standardFeatures=standardFeature,
                                                                        meanValue=meanvalue,
                                                                        standardValue=standardValue)
    return resDict


"""
作用：标准化 file-time-core-fault
返回值：file-time-core-fault的一个字典结构
"""


def standard_file_time_core_faultyDict(ftcPD, standardFeature, meanvalue, standardValue: int):
    resDict = {}
    for filename, time_core_pdDict in ftcPD.items():
        resDict[filename] = {}
        for time, core_pdDict in time_core_pdDict.items():
            resDict[filename][time] = {}
            for icore, faultypdDict in core_pdDict.items():
                resDict[filename][time][icore] = {}
                for ifault, tpd in faultypdDict.items():
                    resDict[filename][time][icore][ifault] = standardPDfromOriginal(tpd,
                                                                                    standardFeatures=standardFeature,
                                                                                    meanValue=meanvalue,
                                                                                    standardValue=standardValue)
    return resDict


# 特征提取 file-time-core数据
def FeaExtra_file_time_core(ftcDict, windowSize: int = 5, windowRealSize: int = 1,
                            silidWindows: bool = True,
                            extraFeature=None):
    resDict = {}
    fault_PDDict = {}
    for filename, time_core_pdDict in ftcDict.items():
        resDict[filename] = {}
        for time, core_pdDict in time_core_pdDict.items():
            resDict[filename][time] = {}
            print("filename:{}-time:{}".format(filename, time))
            for icore, tpd in core_pdDict.items():
                fePD, fault_Dict = featureExtractionUsingFeatures(tpd, windowSize, windowRealSize, silidWindows,
                                                                  extraFeature)
                resDict[filename][time][icore] = fePD
                fault_PDDict = mergeTwoDF(fault_Dict, fault_PDDict)
    return resDict, fault_PDDict


# 特征提取 file-time数据 主要是用于server数据
def FeaExtra_file_time(ftcDict, windowSize: int = 5, windowRealSize: int = 1,
                       silidWindows: bool = True,
                       extraFeature=None):
    resDict = {}
    fault_PDDict = {}
    for filename, time_core_pdDict in ftcDict.items():
        resDict[filename] = {}
        for time, timepd in time_core_pdDict.items():
            fePD, fault_Dict = featureExtractionUsingFeatures(timepd, windowSize, windowRealSize, silidWindows,
                                                              extraFeature)
            resDict[filename][time] = fePD
            fault_PDDict = mergeTwoDF(fault_Dict, fault_PDDict)
            print("filename:{}-time:{}".format(filename, time))
    return resDict, fault_PDDict


# ==================================== 用于server数据
"""
将一个server文件处理的过程
返回值
1. 所有的fault-Pd的一个Dict
2. time-pd 的一个Dict
3. time-fault-pd的一个Dict
"""


def processOneServerFile(spath: str, filepd: pd.DataFrame, accumulationFeatures: List[str],
                         server_features: List[str]):
    if not os.path.exists(spath):
        os.makedirs(spath)

    # 先按照时间段划分
    pdbytime = splitDataFrameByTime(filepd)

    # 将其保存到 spath/1.时间段划分集合
    print("1. 按照时间段划分开始")
    # tmp/{filename}/1.时间段划分集合
    saveDFListToFiles(spath=os.path.join(spath, "1.时间段划分集合文件"), pds=pdbytime)
    print("按照时间段划分结束")
    thistime_pdDict = {}  # time-PD
    thistime_fault_pdDict = {}  # time-fault-PD
    thisFileFaulty_pdDict = {}  # fault-PD

    # 将累计值都减去上一行的
    subcorepds = []
    for i in range(0, len(pdbytime)):
        print("2.{} 第{}个时间段依照核心划分".format(i, i))
        tpd = pdbytime[i]
        tpd = subtractLastLineFromDataFrame(tpd, accumulationFeatures)
        tpd['cpu'] = tpd['user'] + tpd['system']
        tpd = PushLabelToEnd(tpd, FAULT_FLAG)
        subcorepds.append((i, tpd))

    # 将减去上一行的数据进行保存
    tcoresavepath = os.path.join(spath, "2. 时间段划分集合文件-减去上一行")
    saveCoreDFToFiles(tcoresavepath, subcorepds)

    for itime, ipd in subcorepds:
        faultDict = abstractFaultPDDict(ipd, server_features)
        thistime_pdDict[itime] = ipd
        thistime_fault_pdDict[itime] = faultDict
        thisFileFaulty_pdDict = mergeTwoDF(thisFileFaulty_pdDict, faultDict)
    return thisFileFaulty_pdDict, thistime_pdDict, thistime_fault_pdDict


"""
作用：标准化 file-time 
返回值： file-time的字典结构
"""


def standard_file_time_Dict(ftcPD, standardFeature, meanvalue, standardValue: int):
    resDict = {}
    for filename, time_core_pdDict in ftcPD.items():
        resDict[filename] = {}
        for time, timepd in time_core_pdDict.items():
            resDict[filename][time] = standardPDfromOriginal(timepd, standardFeatures=standardFeature,
                                                             meanValue=meanvalue, standardValue=standardValue)
    return resDict


"""
作用：标准化 file-time-fault
返回值：file-time-fault的一个字典结构
"""


def standard_file_time_faultyDict(ftcPD, standardFeature, meanvalue, standardValue: int):
    resDict = {}
    for filename, time_core_pdDict in ftcPD.items():
        resDict[filename] = {}
        for time, core_pdDict in time_core_pdDict.items():
            resDict[filename][time] = {}
            for ifault, faultypd in core_pdDict.items():
                resDict[filename][time][ifault] = standardPDfromOriginal(faultypd, standardFeatures=standardFeature,
                                                                         meanValue=meanvalue,
                                                                         standardValue=standardValue)
    return resDict


"""
遍历所有的核心，得到时间和异常核心数的关系
返回 int - List[int]结构
"""


def getTime_AbnormalCore(ftcPD: Dict):
    tree_time_abnormalCoreDict = defaultdict(list)
    forest_time_abnormalCoreDict = defaultdict(list)
    adapt_time_abnormalCoreDict = defaultdict(list)
    for filename, time_core_pdDict in ftcPD.items():
        for time, core_pdDict in time_core_pdDict.items():
            for icore, tpd in core_pdDict.items():
                # 遍历一整个列表
                for i in range(0, len(tpd)):
                    numiline = tpd.iloc[i]
                    snowtime = numiline[TIME_COLUMN_NAME]
                    inowtime = TranslateTimeToInt(snowtime)
                    # 决策树判断
                    tree_label = MODEL_TYPE[0] + "_flag"
                    tree_pre = numiline[tree_label]
                    if tree_pre != 0:
                        tree_time_abnormalCoreDict[snowtime].append(icore)
                    # 随机森林判断
                    forest_label = MODEL_TYPE[1] + "_flag"
                    forest_pre = numiline[forest_label]
                    if forest_pre != 0:
                        forest_time_abnormalCoreDict[snowtime].append(icore)
                    # 自适应增强判断
                    adapt_label = MODEL_TYPE[2] + "_flag"
                    adapt_pre = numiline[adapt_label]
                    if adapt_pre != 0:
                        adapt_time_abnormalCoreDict[snowtime].append(icore)
    return tree_time_abnormalCoreDict, forest_time_abnormalCoreDict, adapt_time_abnormalCoreDict


"""
- 计算一个开始和结束时间段内的预测结果 将三个字典传输进来 字典是时间-核列表，代表这个时间，这几个核出现了问题
参数说明：
- predictBegintime 开始时间
- predictEndtime   结束时间
- abnormaliTime 是一个异常和时间的集合, 格式如下：
# abnormaliTime = [
#     [c("2021-08-30 13:15:00"), c("2021-08-30 13:36:00"), 31],
#     [c("2021-08-30 13:55:00"), c("2021-08-30 14:16:00"), 32],
#     [c("2021-08-30 14:35:00"), c("2021-08-30 14:56:00"), 33],
#     [c("2021-08-30 15:15:00"), c("2021-08-30 15:36:00"), 34],
#     [c("2021-08-30 15:55:00"), c("2021-08-30 16:16:00"), 35],
#     [c("2021-08-30 16:37:00"), c("2021-08-30 16:56:00"), 11],
# ]
- tree_time_abnormalCoreDict 等其他两个是字典，存储的是时间-核心列表，代表这个时间有哪些核心被检测为异常

备注： 可能会往最后三个字典中添加若干时间，但是其对应的核心列表为空

返回值说明：
返回的是一个pd，存储的是结果说明

"""


def getResultFromTimequantum(predictBegintime: str, predictEndtime: str, abnormaliTime, tree_time_abnormalCoreDict,
                             forest_time_abnormalCoreDict, adapt_time_abnormalCoreDict):
    # 判断一个时间是否属于异常时间段内
    c = TranslateTimeToInt

    def judgeTimeIsAbnormal(nowtime: str, abnormaltimes: List) -> Union[int, Any]:
        inowtime = c(nowtime)
        for i in abnormaltimes:
            if i[0] <= inowtime <= i[1]:
                return i[2]
        return 0

    # 将时间转化为数字， 时间默认格式为 '%Y-%m-%d %H:%M:%S'
    predictBeginitime = TranslateTimeToInt(predictBegintime)
    predictEnditime = TranslateTimeToInt(predictEndtime)
    draw_time_flagDict = {}
    while predictBeginitime <= predictEnditime:
        stime = TranslateTimeToStr(predictBeginitime)
        itime = predictBeginitime
        if stime not in draw_time_flagDict:
            draw_time_flagDict[stime] = {}
        # 真实标签 =====
        flag = judgeTimeIsAbnormal(stime, abnormaliTime)
        draw_time_flagDict[stime][FAULT_FLAG] = flag

        # 下面6行会对参数进行修改
        if stime not in tree_time_abnormalCoreDict:
            tree_time_abnormalCoreDict[stime] = []
        if stime not in forest_time_abnormalCoreDict:
            forest_time_abnormalCoreDict[stime] = []
        if stime not in adapt_time_abnormalCoreDict:
            adapt_time_abnormalCoreDict[stime] = []
        # 决策树标签 =====
        draw_time_flagDict[stime][MODEL_TYPE[0] + "_flag"] = 0
        draw_time_flagDict[stime][MODEL_TYPE[0] + "_num"] = len(tree_time_abnormalCoreDict[stime])
        if len(tree_time_abnormalCoreDict[stime]) != 0:
            draw_time_flagDict[stime][MODEL_TYPE[0] + "_flag"] = 30
        # 随机森林标签 ====
        draw_time_flagDict[stime][MODEL_TYPE[1] + "_flag"] = 0
        draw_time_flagDict[stime][MODEL_TYPE[1] + "_num"] = len(forest_time_abnormalCoreDict[stime])
        if len(forest_time_abnormalCoreDict[stime]) != 0:
            draw_time_flagDict[stime][MODEL_TYPE[1] + "_flag"] = 30

        # 自适应增强标签 ====
        draw_time_flagDict[stime][MODEL_TYPE[2] + "_flag"] = 0
        draw_time_flagDict[stime][MODEL_TYPE[2] + "_num"] = len(adapt_time_abnormalCoreDict[stime])
        if len(adapt_time_abnormalCoreDict[stime]) != 0:
            draw_time_flagDict[stime][MODEL_TYPE[2] + "_flag"] = 30
        predictBeginitime += 60

    time_coreinformationtpd = pd.DataFrame(data=draw_time_flagDict).T
    time_coreinformationtpd = time_coreinformationtpd.reset_index()
    time_coreinformationtpd = time_coreinformationtpd.rename(columns={"index": "time"})
    return time_coreinformationtpd
