import itertools
import json
import os.path
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Any, Set

import numpy as np
import pandas as pd
from pandas import DataFrame

from Classifiers.ModelPred import select_and_pred
from utils.DataFrameOperation import mergeDataFrames, SortLabels, PushLabelToFirst, PushLabelToEnd, \
    subtractLastLineFromDataFrame
from utils.DataScripts import getDFmean, standardPDfromOriginal, TranslateTimeListStrToStr, standardPDfromOriginal1
from utils.DefineData import TIME_COLUMN_NAME, FAULT_FLAG, PID_FEATURE, CPU_FEATURE, MODEL_TYPE
from utils.FileSaveRead import saveDFListToFiles


def getfilepd(ipath: str) -> pd.DataFrame:
    if not os.path.exists(ipath):
        filename = os.path.basename(ipath)
        print("{} 文件不存在".format(filename))
        exit(1)
    tpd = pd.read_csv(ipath)
    return tpd


"""
将含有特征的值 去掉time和faultFlag
"""


def getuserfulfeature(clistnames: List[str]):
    if FAULT_FLAG in clistnames:
        clistnames.remove(FAULT_FLAG)
    if TIME_COLUMN_NAME in clistnames:
        clistnames.remove(TIME_COLUMN_NAME)
    return clistnames


# 将一个列表里面的pds进行标准化并且作为一个列表进行返回
def standardLists(pds: List[pd.DataFrame], standardFeatures: List[str], meanValue, standardValue: int = 100) -> List[
    pd.DataFrame]:
    standardList = []
    for ipd in pds:
        tpd = standardPDfromOriginal1(ipd, standardFeatures, meanValue, standardValue)
        standardList.append(tpd)
    return standardList


# 对进程的pdlists中的每个pd计算cpu的数值
def add_cpu_column(pds: List[pd.DataFrame]) -> List[pd.DataFrame]:
    for ipd in pds:
        ipd['cpu'] = ipd['user'] + ipd['system']
    return pds


"""
将时间都转化为标准格式
处理完之后的格式为这种形式
'%Y-%m-%d %H:%M:00'
"""


# 将一个pd中的时间序列的秒变为0
def changeTimeColumns(df: pd.DataFrame, timeformat: str = '%Y/%m/%d %H:%M') -> pd.DataFrame:
    tpd = df.loc[:, [TIME_COLUMN_NAME]].apply(lambda x: TranslateTimeListStrToStr(x.to_list(), timeformat), axis=0)
    df.loc[:, TIME_COLUMN_NAME] = tpd.loc[:, TIME_COLUMN_NAME]
    return df


# 讲一个pd的列表全都改变
def changeTimeTo_pdlists(pds: List[pd.DataFrame], timeformat: str = '%Y/%m/%d %H:%M') -> List[pd.DataFrame]:
    changed_pds = []
    for ipd in pds:
        tpd = changeTimeColumns(ipd, timeformat)
        changed_pds.append(tpd)
    return changed_pds


"""
特征提取一个dataframe
愿数据保持变就是会添加一些新的特征
假设滑动窗口为3， 那么将会舍弃开始的2个窗口，
"""


def featureExtractionPd(df: pd.DataFrame, extraFeature: List[str], windowSize: int = 5) -> pd.DataFrame:
    # 一个用于判断x的百分比的值
    def quantile(n):
        def quantile_(x):
            return np.quantile(x, q=n)

        quantile_.__name__ = 'quantile_%d' % (n * 100)
        return quantile_

    # 得到的这个DataFrame是一个二级列名类似下面这种
    #	    system	                            ｜                      user
    #  sum	mean	amax	amin	quantile50  ｜ sum	mean	amax	amin	quantile50
    featureExtractionDf = df.loc[:, extraFeature].rolling(window=windowSize, min_periods=1).agg(
        [np.mean, np.max, np.min, quantile(0.5)])

    # 将二级索引变成一级的
    featureExtractionDf.columns = ["_".join(x) for x in featureExtractionDf.columns]

    # 此时应该将其和原本的df进行合并，这样既保存了原文件，也保存了新的数据
    resdf = pd.concat([df, featureExtractionDf], axis=1)
    resdf = resdf.dropna()

    # 标签进行排序
    resdf = SortLabels(resdf)
    resdf = PushLabelToFirst(resdf, TIME_COLUMN_NAME)
    resdf = PushLabelToEnd(resdf, FAULT_FLAG)
    resdf.dropna(inplace=True)
    resdf.reset_index(drop=True, inplace=True)
    return resdf


"""
函数功能： 处理一个DataFrame的数据，每个数据来自一个单独的文件，返回标准化提取之后的一个DataFrame
1. 根据pid进行分割, 必须包含pid这个列名
返回特征处理之后的一个DataFrame

参数说明：
extractFeatures : 要提取的特征 
accumulateFeatures : 累计特征 都在extractFeatures

"""


def processpd_bypid(processpd: pd.DataFrame, extractFeatures: List[str], accumulateFeatures: List[str],
                    windowsSize: int = 3, spath: str = None) -> pd.DataFrame:
    if spath is not None and not os.path.exists(spath):
        os.makedirs(spath)
    pidpds = []
    print(PID_FEATURE.center(40, "*"))
    for ipid, idf in processpd.groupby(PID_FEATURE):
        print("pid: {} ".format(ipid), end="")
        idf: pd.DataFrame
        # 对每一个进程开始的前两个点和后两个点都去掉
        assert len(idf) > 4
        idf = idf.iloc[2:-2]
        print("size: {}".format(idf.size))
        # 进行累计差分处理
        # subtractpd = subtractLastLineFromDataFrame(idf, columns=accumulateFeatures)
        # 对对应的特征进行提取
        featureExtractionDf = featureExtractionPd(idf, extraFeature=extractFeatures, windowSize=windowsSize)
        # 将特征提取之后的效果进行保存
        if spath is not None:
            featureExtractionDf.to_csv(os.path.join(spath, "{}.csv".format(ipid)))
        pidpds.append(featureExtractionDf)
    allpidpds, _ = mergeDataFrames(pidpds)
    return allpidpds


"""
将所有的process数据进行特征提取
"""


def processpdsList(processpds: List[pd.DataFrame], extractFeatures: List[str], accumulateFeatures: List[str],
                   windowsSize: int = 3, spath: str = None) -> List[pd.DataFrame]:
    featureExtractiondfs = []
    for ipd in processpds:
        tpd = processpd_bypid(ipd, extractFeatures, accumulateFeatures, windowsSize, spath)
        featureExtractiondfs.append(tpd)
    return featureExtractiondfs


"""
将所有的server数据进行特征提取
参数说明：
extractFeatures : 要提取的特征 
accumulateFeatures : 累计特征 都在extractFeatures

"""


def serverpdsList(serverpds: List[pd.DataFrame], extractFeatures: List[str],
                  windowsSize: int = 3, spath: str = None) -> List[pd.DataFrame]:
    if spath is not None and not os.path.exists(spath):
        os.makedirs(spath)
    extraction_dfs = []
    for i, iserverpd in enumerate(serverpds):
        # 对累计的特征值进行数据的处理, 默认一个server数据里面都是连续的, 就算不连续，也只会影响几个点
        # subtractpd = subtractLastLineFromDataFrame(iserverpd, columns=accumulateFeatures)
        # 对特征值进行特征提取
        featureExtractionDf = featureExtractionPd(iserverpd, extraFeature=extractFeatures, windowSize=windowsSize)
        if spath is not None:
            featureExtractionDf.to_csv(os.path.join(spath, "server" + str(i) + ".csv"))
        extraction_dfs.append(featureExtractionDf)
    return extraction_dfs


"""
通过阀值判断 一个predictpd中每一行是不是CPU异常，如果是CPU异常，就返回True，如果是正常就返回Falase

"""


def isCPUAbnormalsByThreshold(predictpd: pd.DataFrame, thresholdValue: Dict) -> bool:
    # 主要是通过predictpd中的cpu_mean指标进行判断, 如果认为其小于等于阀值，就为True，否则就为faluse
    cpumean_feature = "cpu_mean"
    predictpd.reset_index(inplace=True, drop=True)
    # 阀值cpu的平均值, key要保持一致
    process_cpu_meanvalue = thresholdValue["process_cpu_mean"]
    predictpd: pd.DataFrame
    reslist = predictpd.loc[:, cpumean_feature] <= process_cpu_meanvalue
    return reslist


"""
得到这一个一个processpd对应的时间下，所有的cputime，以及通过预测有那几个核心是cpu异常
判断是否使用阀值, 如果使用阀值， 那么阀值的大小都在thtrsholdValue中存储, 如果不使用阀值，那么modelfilepath就不为空
返回值： 1. 异常的数量列表  2. 异常的核心列表 3. 异常的cpu最小值列表
"""


def getprocess_cputime_abcores(processpds: pd.DataFrame, nowtime: str, isThreshold: bool = False,
                               thresholdValue: Dict = None, modelfilepath: str = None) -> Union[
    tuple[int, None, None], tuple[Any, list, list]]:
    nowdf = processpds[processpds[TIME_COLUMN_NAME] == nowtime]
    if len(nowdf) == 0:
        return 0, None, None

    cpu_mean_fe = "cpu_mean"
    # 先得到总的CPUTIME的时间
    cputime = nowdf[cpu_mean_fe].sum()
    # 核的编号
    cores_serialnumber = list(nowdf.loc[:, CPU_FEATURE])
    # 核的cpu时间
    cores_max_time = list(nowdf.loc[:, cpu_mean_fe])
    if isThreshold:
        predictflag = isCPUAbnormalsByThreshold(nowdf, thresholdValue)
    else:
        predictflag = select_and_pred(nowdf, MODEL_TYPE[0], saved_model_path=modelfilepath)
        predictflag = [True if i != 0 else False for i in predictflag]
    # predictflag为True代表异常， 否则代表这正常
    # 获得异常的核
    assert len(predictflag) == len(cores_serialnumber)
    abnormalcores = [cores_serialnumber[i] for i, flag in enumerate(predictflag) if flag]
    abnormalcoremaxtime = [cores_max_time[i] for i, flag in enumerate(predictflag) if flag]
    # 将所有的cputime和不正常的核心数据进行返回
    return cputime, abnormalcores, abnormalcoremaxtime


"""
将server文件和process结合，根据时间对数据进行分析，最后得到一个Dict，包含如下信息
time, server_flag(可选), used, used_mean, pgfree, pgfree_mean, pgfree_min, pgfree_max, wrf_cpu, abnormalcore(是一个列表)
返回值key 
time    wrf_cpu    abnormalcores    faultyFlag   "used", "used_mean", "pgfree", "pgfree_mean", "pgfree_amin", "pgfree_amax"  coresnums
serverinformationDict: 
abnormalcores 是一个列表的列表，存储的是异常的核心数
coresnums 是指多少个核心出现了异常
coresmaxtime: 是一个列表的列表， 存储的是每个异常核心的值
"""


def deal_serverpds_and_processpds(allserverpds: pd.DataFrame, allprocesspds: pd.DataFrame, spath: str = None,
                                  isThreshold: bool = False, thresholdValue: Dict = None,
                                  modelfilepath: str = None) -> Dict:
    if spath is not None and not os.path.exists(spath):
        os.makedirs(spath)
    # 将allserverpds里面所有的时间搜集起来
    timecolumns = allserverpds[TIME_COLUMN_NAME]
    serverinformationDict = defaultdict(list)
    for stime in timecolumns:
        # 添加wrf的cpu时间
        wrf_cpu_time, abnormalcores, abnormalcoremaxtime = getprocess_cputime_abcores(allprocesspds, stime, isThreshold=isThreshold,
                                                                 thresholdValue=thresholdValue,
                                                                 modelfilepath=modelfilepath)
        # 不管返回值如何都进行直接的添加
        serverinformationDict["wrf_cpu"].append(wrf_cpu_time)
        serverinformationDict["abnormalcores"].append(abnormalcores)
        serverinformationDict["coresmaxtime"].append(abnormalcoremaxtime)
    # 将server_flag加入进来, 假如存在的话
    if FAULT_FLAG in allserverpds.columns.array:
        serverinformationDict[FAULT_FLAG] = list(allserverpds[FAULT_FLAG])
    add_server_feature = ["time", "used", "used_mean", "pgfree", "pgfree_mean", "pgfree_amin", "pgfree_amax"]
    for ife in add_server_feature:
        serverinformationDict[ife] = list(allserverpds[ife])

    # 得到cpu每个时间点的数量
    coresnumslist = [len(i) if i is not None else -1 for i in serverinformationDict["abnormalcores"]]
    serverinformationDict["coresnums"] = coresnumslist

    # 加入一个断言
    assert len(serverinformationDict["wrf_cpu"]) == len(allserverpds)

    # 将字典中的数据进行保存 ==========================================================================================
    nosavefeatures = ["abnormalcores"]
    savedict = dict(
        [(key, serverinformationDict[key]) for key in serverinformationDict.keys() if key not in nosavefeatures])
    savedict["coresnums"] = coresnumslist
    tpd = pd.DataFrame(data=savedict)
    tpd.to_csv(os.path.join(spath, "server_process有用指标.csv"))
    # 将时间和错误核心列表进行保存
    timelist = serverinformationDict[TIME_COLUMN_NAME]
    abcores = serverinformationDict["abnormalcores"]  # 是一个列表的集合
    assert len(timelist) == len(abcores)

    time_abcoresDict = dict(zip(timelist, abcores))
    time_abcoresJson = json.dumps(time_abcoresDict, indent=4, sort_keys=True)
    with open(os.path.join(spath, "time_abnormalCores.json"), "w") as f:
        f.write(str(time_abcoresJson))
    # ==============================================================================================================
    return serverinformationDict


# ============================================================================================================== 预测信息

"""
对CPU进行预测，
返回值是一个列表, 0 代表这个时间段预测为正常，1预测为CPU异常, -1代表是边界，无法进行预测
预测标准是：
10 代表全CPU异常
20 代表单CPU抢占
30 代表多CPU抢占
80 代表随机抢占
-1 代表这个时间没有数据
"""


def predictcpu(serverinformationDict: Dict, coresnumber: int = 0) -> List[int]:
    #  wrfnumList不为None
    wrfnumList = serverinformationDict['abnormalcores']
    assert len(serverinformationDict[TIME_COLUMN_NAME]) == len(wrfnumList)
    iscpu = []  # 先添加一个数值，最后返回的时候要去掉
    ilastlist = None
    for i, ilist in enumerate(wrfnumList):
        # ===========================
        if ilist is None:
            iscpu.append(-1)
            ilastlist = None
            continue
        # ========================
        if len(ilist) == 0:
            iscpu.append(0)
            ilastlist = []
            continue
        # ========================
        if len(ilist) == 1:
            if ilastlist is None:
                iscpu.append(20)
            elif len(ilastlist) == 0:
                iscpu.append(20)
            elif len(ilastlist) == 1 and set(ilastlist) == set(ilist):
                iscpu.append(20)
            elif len(ilastlist) == 1 and set(ilastlist) != set(ilist):
                iscpu[-1] = 80
                iscpu.append(80)
            elif len(ilastlist) > 1:
                iscpu[-1] = 80
                iscpu.append(80)
            else:
                print("len(list) == 1: 预测cpu出现了不可预知的错误")
                exit(1)
            ilastlist = ilist
            continue
        # =======================
        if len(ilist) == coresnumber:
            if ilastlist is None:
                iscpu.append(10)
            elif len(ilastlist) == 0:
                iscpu.append(10)
            elif len(ilastlist) == coresnumber:
                iscpu.append(10)
            else:
                iscpu[-1] = 80
                iscpu.append(80)
            ilastlist = ilist
            continue
        # =======================
        # 现在就是多核心cpu的数据
        if ilastlist is None:
            iscpu.append(30)
        elif len(ilastlist) == 0:
            iscpu.append(30)
        elif len(ilastlist) == 1:
            iscpu[-1] = 80
            iscpu.append(80)
        elif len(ilastlist) == coresnumber:
            iscpu[-1] = 80
            iscpu.append(80)
        elif len(ilastlist) != len(ilist):
            iscpu[-1] = 80
            iscpu.append(80)
        elif len(ilastlist) == len(ilist) and set(ilastlist) != set(ilist):
            iscpu[-1] = 80
            iscpu.append(80)
        elif len(ilastlist) == len(ilist) and set(ilastlist) == set(ilist):
            iscpu[-1] = 30
            iscpu.append(30)
        else:
            print("多核cpu 来到了不可能来到的位置")
            exit(1)
        ilastlist = ilist
    return iscpu


"""
对内存泄露进行预测
有两种方式，一种是通过阀值，一种是通过模型， 如果是通过模型则需要生成一个dataFrame传递进去
"""


def predict_memory_leaks(serverinformationDict: Dict, isThreshold: bool = False, thresholdValue: Dict = None,
                         Memory_leaks_modelpath: str = None) -> List:
    if isThreshold:
        memoryleakValue = thresholdValue["used"]
        realmemoryleakValue = serverinformationDict["used_mean"]
        prelistflag = [60 if i > memoryleakValue else 0 for i in realmemoryleakValue]
    else:
        # 先构造一个字典，然后生成dataFrame, 调用接口进行预测
        used_features = ["used_mean", "used_amax", "used_amin"]
        savedict = dict(
            [(key, serverinformationDict[key]) for key in serverinformationDict.keys() if key in used_features])
        tpd = pd.DataFrame(data=savedict)
        prelistflag = select_and_pred(tpd, MODEL_TYPE[0], saved_model_path=Memory_leaks_modelpath)

    return prelistflag


"""
对内存带宽进行预测
有两种方式，一种是通过阀值，一种是通过模型， 如果是通过模型则需要生成一个dataFrame传递进去
"""


def predict_memory_bandwidth(serverinformationDict: Dict, isThreshold: bool = False, thresholdValue: Dict = None,
                             Memory_bandwidth_modelpath: str = None) -> List:
    if isThreshold:
        memorybandwidthValue = thresholdValue["pgfree"]
        realmemorywidthValue = serverinformationDict["pgfree_mean"]
        prelistflag = [50 if i > memorybandwidthValue else 0 for i in realmemorywidthValue]
    else:
        # 先构造一个字典，然后生成dataFrame, 调用接口进行预测
        used_features = ["pgfree_mean", "pgfree_amax", "pgfree_amin"]
        savedict = dict(
            [(key, serverinformationDict[key]) for key in serverinformationDict.keys() if key in used_features])
        tpd = pd.DataFrame(data=savedict)
        prelistflag = select_and_pred(tpd, MODEL_TYPE[0], saved_model_path=Memory_bandwidth_modelpath)

    return prelistflag


"""
根据CPU异常，内存泄露异常以及多CPU异常判断
coresList代表每个时间段

这个预测会将孤立点去除
"""


def get_realpredict(predictDict: Dict, coresList: List) -> List:
    cpu_list = predictDict["CPU_Abnormal"]
    leak_list = predictDict["mem_leak"]
    bandwidth_list = predictDict["mem_bandwidth"]
    assert len(cpu_list) == len(coresList)

    preflag = []
    for i in range(0, len(cpu_list)):
        # 如果coresList[i]为null， 代表这个时间点，在process中不存在，第一种可能是：真的不存在  第二种可能是: 作为进程一开始，我将刚刚开始的两分钟和结尾的两分钟删除了
        if coresList[i] is None:
            preflag.append(0)
            continue
        if bandwidth_list[i] != 0:
            preflag.append(50)
            continue
        if leak_list[i] != 0:
            preflag.append(60)
            continue
        preflag.append(cpu_list[i])
    # 去除那些孤立的点，如 00 异常 00中的异常 或者 1 1 0 1 1 中的正常
    for i in range(2, len(preflag) - 2):
        if preflag[i - 2] != 0 and preflag[i - 1] != 0 and preflag[i] == 0 and preflag[i + 1] != 0 and preflag[i + 2] != 0:
            preflag[i] = preflag[i - 1] # 如果遇到这种情况，取得上一个数
        if preflag[i - 2] == 0 and preflag[i - 1] == 0 and preflag[i] != 0 and preflag[i + 1] == 0 and preflag[i + 2] == 0:
            preflag[i] = 0
    return preflag


def getSingleMaxCPUTime(serverinformationDict: Dict) -> List[int]:
    # 某一时刻的cpu列表
    wrfnumList = serverinformationDict['coresmaxtime']
    singlemintime = []
    for itimecores in wrfnumList:
        if itimecores is None:
            singlemintime.append(-1)
            continue
        if len(itimecores) == 0:
            singlemintime.append(0)
            continue
        else:
            singlemintime.append(max(itimecores))
            continue
    return singlemintime




"""
将得到的基本信息都得到之后，对结果进行分析
如果isThreshold = True 那么就使用阀值预测，否则就使用模型预测，这个时候，模型路径不能为空
coresnumber设置一下，主要是预测全CPU抢占
serverinformationDict: 
abnormalcores 是一个列表的列表，存储的是异常的核心数
coresnums 是指多少个核心出现了异常
coresmaxtime: 是一个列表的列表， 存储的是每个异常核心的值
"""


def predictAllAbnormal(serverinformationDict: Dict, spath: str, isThreshold: bool = False,
                       thresholdValue: Dict = None,
                       Memory_bandwidth_modelpath: str = None, Memory_leaks_modelpath: str = None,
                       coresnumber: int = 0) -> pd.DataFrame:
    predictDict = {}
    predictDict[TIME_COLUMN_NAME] = serverinformationDict[TIME_COLUMN_NAME]
    if FAULT_FLAG in serverinformationDict.keys():
        predictDict[FAULT_FLAG] = serverinformationDict[FAULT_FLAG]
    # 对CPU进行预测
    predictDict["CPU_Abnormal"] = predictcpu(serverinformationDict, coresnumber)
    # 对内存泄露进行预测
    predictDict["mem_leak"] = predict_memory_leaks(
        serverinformationDict=serverinformationDict,
        isThreshold=isThreshold,
        thresholdValue=thresholdValue,
        Memory_leaks_modelpath=Memory_leaks_modelpath
    )
    # 对内存带宽进行预测
    predictDict["mem_bandwidth"] = predict_memory_bandwidth(
        serverinformationDict=serverinformationDict,
        isThreshold=isThreshold,
        thresholdValue=thresholdValue,
        Memory_bandwidth_modelpath=Memory_bandwidth_modelpath
    )
    # 得到核的数量
    predictDict["coresnums"] = serverinformationDict["coresnums"]
    # 根据CPU信息和得到真是标签值
    predictDict["preFlag"] = get_realpredict(predictDict, serverinformationDict["abnormalcores"])
    # 某一时刻的cpu列表
    wrfnumList = serverinformationDict['abnormalcores']
    # 得到某一时刻下
    predictDict["smincputime"] = getSingleMaxCPUTime(serverinformationDict)


    tpd = pd.DataFrame(data=predictDict)
    tpd = PushLabelToFirst(tpd, "preFlag")
    tpd = PushLabelToFirst(tpd, FAULT_FLAG)
    tpd = PushLabelToFirst(tpd, TIME_COLUMN_NAME)
    # ====================将结果进行保存
    if not os.path.exists(spath):
        os.makedirs(spath)
    tpd.to_csv(os.path.join(spath, "预测结果.csv"))
    return tpd


def getfilespath(filepath: str) -> List[str]:
    if not os.path.exists(filepath):
        print("{}路径不存在".format(filepath))
        exit(1)
    files = os.listdir(filepath)
    filepaths = [os.path.join(filepath, i) for i in files]
    return filepaths


"""
对进程数据进行差分, 其他数据全部保存不变
进程数据的差分是按照pid进行修改的
"""


def differenceProcess(processpds: List[pd.DataFrame], accumulateFeatures: List[str]) -> List[pd.DataFrame]:
    differencepds = []
    for iprocesspd in processpds:
        subtractpdLists = []
        for ipid, ipd in iprocesspd.groupby(PID_FEATURE):
            # 先将一些不可用的数据进行清除,比如一个进程只运行了两分钟
            if len(ipd) <= 1:
                continue
            subtractpd = subtractLastLineFromDataFrame(ipd, columns=accumulateFeatures)
            subtractpdLists.append(subtractpd)
        allsubtractpd, _ = mergeDataFrames(subtractpdLists)
        differencepds.append(allsubtractpd)
    return differencepds

"""
对数据进行差分处理
并且对pgfree这个指标进行中位数平滑
"""
def differenceServer(serverpds: List[pd.DataFrame], accumulateFeatures: List[str]) -> List[pd.DataFrame]:
    differencepds = []
    for iserverpd in serverpds:
        subtractpd = subtractLastLineFromDataFrame(iserverpd, columns=accumulateFeatures)
        differencepds.append(subtractpd)
    # 中位数进行平滑操作
    differencepds = smooth_pgfree(differencepds, smoothwinsize=7)
    return differencepds


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
def removeAllHeadTail(predictPd: pd.DataFrame, windowsize: int = 3) -> pd.DataFrame:
    allabnormals = set(predictPd[FAULT_FLAG])
    if 0 in allabnormals:
        allabnormals.remove(0)
    removepd = removeHeadTail_specifiedAbnormal(predictPd, windowsize=windowsize, abnormals=allabnormals)
    return removepd


# 去除指定异常及其首尾数据
def remobe_Abnormal_Head_Tail(predictPd: pd.DataFrame, abnormals: Set[int], windowsize: int = 3) -> pd.DataFrame:
    dealflag = "faultFlag"

    def judge(x: pd.Series):
        # abnormals中有一个
        if len(abnormals & set(x)) != 0:
            return False  # 表示去除
        else:
            return True  # 表示保留

    savelines = predictPd[dealflag].rolling(window=windowsize, min_periods=1).agg([judge])["judge"].astype("bool")
    return predictPd[savelines]


# 计算调和平均数
def harmonic_mean(data):  # 计算调和平均数
    total = 0
    for i in data:
        if i == 0:  # 处理包含0的情况
            return 0
        total += 1 / i
    return len(data) / total

"""
得到时间段的信息
返回实际预测时间段的个数  预测时间段的个数  以及相交的时间段个数
"""
def getTimePeriodInfo(predictpd: pd.DataFrame):
    timeslist = predictpd[TIME_COLUMN_NAME]
    realflag = predictpd[FAULT_FLAG]
    preflag = predictpd["preFlag"]
    assert len(realflag) == len(preflag) # 断言
    # 得到realflag异常的时间段
    tlist1 = [False if i == 0 else True for i in realflag]
    tlist = [k for k, g in itertools.groupby(tlist1)] # 去除重复
    realperiodLen = len([i for i in tlist if i]) # 找到异常的数值

    # 得到preflag异常的时间段
    tlist2 = [False if i == 0 else True for i in preflag]
    tlist = [k for k, g in itertools.groupby(tlist)]
    preperiodLen = len([i for i in tlist if i])

    #
    sameLen = 0
    for i in range(0, len(tlist) - 3):
        if tlist1[i] and tlist1[i + 1] and tlist1[i + 2] and tlist2[i] and tlist[i + 1] and tlist[i + 2]:
            sameLen += 1

    return realperiodLen, preperiodLen, sameLen



"""
将abnormalsList中的异常当做同一种类的异常, 其预测
"""

def getBasicInfo(predictpd: pd.DataFrame, abnormalsSet: Set) -> Dict:
    infoDict = {}
    # num 数据的数量
    # recall  召回率
    # precision 精确率
    # per_normal 预测为正常的百分比
    # per_fault 预测为异常的百分比
    preflaglabel = "preFlag"
    realflags = list(predictpd[FAULT_FLAG])
    preflags = list(predictpd[preflaglabel])
    assert len(realflags) == len(preflags)
    rightflagSet = set([(i // 10) * 10 for i in abnormalsSet])  # 如果预测在这个集合中， 则认为预测正确

    real_abnormalnums = 0 # 异常的总数量
    pre_allabnormalnums = 0 # 所有预测数据中，被预测为异常的数量
    abnormal_rightabnormal_nums = 0 # 异常被预测为正确的个数
    abnormal_abnormal_nums = 0 # 异常被预测为!=0的数量
    abnormal_normal_nums = 0 # 异常被预测为正常的数量

    for i in range(len(realflags)):
        if realflags[i] in abnormalsSet:
            real_abnormalnums += 1 # 表示异常的真实数量+1
        if preflags[i] in rightflagSet:
            pre_allabnormalnums += 1 # 被预测为异常的真实数量+1

        if realflags[i] in abnormalsSet:
            # 现在实际预测值是异常
            if preflags[i] == 0:
                # 预测值是0
                abnormal_normal_nums += 1
            if preflags[i] != 0:
                abnormal_abnormal_nums += 1
            if preflags[i] in rightflagSet:
                abnormal_rightabnormal_nums += 1 # 异常预测正确

    infoDict["num"] = real_abnormalnums
    infoDict["recall"] = -1 if real_abnormalnums == 0 else abnormal_rightabnormal_nums / real_abnormalnums
    infoDict["precison"] = -1 if pre_allabnormalnums == 0  else abnormal_rightabnormal_nums / pre_allabnormalnums
    infoDict["pre_abnormal"] = -1 if real_abnormalnums == 0 else abnormal_abnormal_nums / real_abnormalnums # 预测为异常的比例, 异常的发现率
    infoDict["pre_normal"] = -1 if real_abnormalnums == 0 else abnormal_normal_nums / real_abnormalnums # 预测为正常的比例
    infoDict["f-score"] =  harmonic_mean([infoDict["recall"], infoDict["precison"]])
    return infoDict


# time  faultFlag  preFlag  mem_leak  mem_bandwidth
# 主要分析三种情况，1. 不去除首位的，2. 去除首位  3. 去除低等级
# 得到10 20 30 50 60 以及 将10 20 30当作cpu 一种情况
def analysePredictResult(predictpd: pd.DataFrame, spath: str, windowsize:  int = 3):
    # 先将{40, 70, 90} 这三种异常去除,并且去除其首尾数据
    predictpd = remobe_Abnormal_Head_Tail(predictpd, windowsize=windowsize, abnormals={
        41, 42, 43, 44, 45,
        71, 72, 73, 74, 75,
        91, 92, 93, 94, 95
    })

    # 预测不去除首位数据 ===============================================================================================
    analyseDict = {}
    analyseDict[0] = getBasicInfo(predictpd, {0})
    analyseDict[10] = getBasicInfo(predictpd, {11, 12, 13, 14, 15})
    analyseDict[20] = getBasicInfo(predictpd, {21, 22, 23, 24, 25})
    analyseDict[30] = getBasicInfo(predictpd, {31, 32, 33, 34, 35})
    analyseDict[50] = getBasicInfo(predictpd, {51, 52, 53, 54, 55})
    analyseDict[60] = getBasicInfo(predictpd, {61, 62, 63, 64, 65})
    analyseDict[80] = getBasicInfo(predictpd, {81, 82, 83, 84, 85})
    analyseDict["cpu"] = getBasicInfo(predictpd, {
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35,
        81, 82, 83, 84, 85
    })
    analyseDict["memory"] = getBasicInfo(predictpd, {
        51, 52, 53, 54, 55,
        61, 62, 63, 64, 65
    })
    # 将信息进行保存
    tpd = pd.DataFrame(data=analyseDict).T
    tpd.to_csv(os.path.join(spath, "1. 不去除首位_统计数据.csv"))

    # 预测全部异常去除首尾之后的数据 ==================================================================================
    tpd = removeAllHeadTail(predictPd=predictpd, windowsize=windowsize)
    analyseDict = {}
    analyseDict[0] = getBasicInfo(tpd, {0})
    analyseDict[10] = getBasicInfo(tpd, {11, 12, 13, 14, 15})
    analyseDict[20] = getBasicInfo(tpd, {21, 22, 23, 24, 25})
    analyseDict[30] = getBasicInfo(tpd, {31, 32, 33, 34, 35})
    analyseDict[50] = getBasicInfo(tpd, {51, 52, 53, 54, 55})
    analyseDict[60] = getBasicInfo(tpd, {61, 62, 63, 64, 65})
    analyseDict[80] = getBasicInfo(tpd, {81, 82, 83, 84, 85})
    analyseDict["cpu"] = getBasicInfo(tpd, {
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35,
        81, 82, 83, 84, 85
    })
    analyseDict["memory"] = getBasicInfo(tpd, {
        51, 52, 53, 54, 55,
        61, 62, 63, 64, 65
    })
    # ===============================================================================统计时间段信息
    realperiodLen, preperiodLen, sameLen = getTimePeriodInfo(tpd)
    writeinfo = ["实际时间段个数: {}".format(realperiodLen), "预测时间段个数：{}".format(preperiodLen), "预测准确时间段个数: {}".\
        format(sameLen), "预测召回率: {:.2%}".format(sameLen / realperiodLen), "预测精确率: {:.2%}".format(sameLen / preperiodLen)]
    # ===============================================================================
    # 将信息进行保存
    tpd = pd.DataFrame(data=analyseDict).T
    tpd.to_csv(os.path.join(spath, "2. 去除首位_统计数据.csv"))

    with open(os.path.join(spath, "2. 去除首位_统计信息时间段.txt", "w")) as f:
        f.writelines(writeinfo)

    # 预测去除低等级之后的数据 包括首尾数据===============================================================================
    tpd = remobe_Abnormal_Head_Tail(predictPd=predictpd, windowsize=windowsize, abnormals={
        11, 12,
        21, 22,
        31, 32,
        51, 52,
        61, 62,
        81, 82
    })
    # 去除所有的首尾数据
    tpd = removeAllHeadTail(tpd, windowsize=windowsize)
    analyseDict = {}
    analyseDict[0] = getBasicInfo(tpd, {0})
    analyseDict[10] = getBasicInfo(tpd, {11, 12, 13, 14, 15})
    analyseDict[20] = getBasicInfo(tpd, {21, 22, 23, 24, 25})
    analyseDict[30] = getBasicInfo(tpd, {31, 32, 33, 34, 35})
    analyseDict[50] = getBasicInfo(tpd, {51, 52, 53, 54, 55})
    analyseDict[60] = getBasicInfo(tpd, {61, 62, 63, 64, 65})
    analyseDict[80] = getBasicInfo(tpd, {81, 82, 83, 84, 85})
    analyseDict["cpu"] = getBasicInfo(tpd, {
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35,
        81, 82, 83, 84, 85
    })
    analyseDict["memory"] = getBasicInfo(tpd, {
        51, 52, 53, 54, 55,
        61, 62, 63, 64, 65
    })
    # 将信息进行保存
    tpd = pd.DataFrame(data=analyseDict).T
    tpd.to_csv(os.path.join(spath, "3. 去除首位_去除低强度_统计数据.csv"))
    # ==================================================================================================

"""
对server数据列表中pgfree进行滑动窗口的处理
"""
def smooth_pgfree(serverpds: List[pd.DataFrame], smoothwinsize: int = 6) -> List[pd.DataFrame]:
    pgfree_name = "pgfree"
    for ipd in serverpds:
        if pgfree_name in ipd.columns.array:
            ipd[pgfree_name] = ipd[pgfree_name].rolling(window=smoothwinsize, min_periods=1, center=True).median()
    return serverpds






