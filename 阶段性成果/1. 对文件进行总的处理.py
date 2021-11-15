import json
import os.path
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from Classifiers.ModelPred import select_and_pred
from utils.DataFrameOperation import mergeDataFrames, SortLabels, PushLabelToFirst, PushLabelToEnd, \
    subtractLastLineFromDataFrame
from utils.DataScripts import getDFmean, standardPDfromOriginal, TranslateTimeListStrToStr
from utils.DefineData import TIME_COLUMN_NAME, FAULT_FLAG, PID_FEATURE, CPU_FEATURE, MODEL_TYPE


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
        tpd = standardPDfromOriginal(ipd, standardFeatures, meanValue, standardValue)
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
    featureExtractionDf = df.loc[:, extraFeature].rolling(window=windowSize).agg(
        [np.mean, np.max, np.min, quantile(0.5)])

    # 将二级索引变成一级的
    featureExtractionDf.columns = ["_".join(x) for x in featureExtractionDf.columns.ravel()]

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
        idf: pd.DataFrame
        print("pid: {} size: {}".format(ipid, idf.size))
        # 进行累计差分处理
        subtractpd = subtractLastLineFromDataFrame(idf, columns=accumulateFeatures)
        # 对对应的特征进行提取
        featureExtractionDf = featureExtractionPd(subtractpd, extraFeature=extractFeatures, windowSize=windowsSize)
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


def serverpdsList(serverpds: List[pd.DataFrame], extractFeatures: List[str], accumulateFeatures: List[str],
                  windowsSize: int = 3, spath: str = None) -> List[pd.DataFrame]:
    if spath is not None and not os.path.exists(spath):
        os.makedirs(spath)
    extraction_dfs = []
    for i, iserverpd in range(serverpds):
        # 对累计的特征值进行数据的处理, 默认一个server数据里面都是连续的, 就算不连续，也只会影响几个点
        subtractpd = subtractLastLineFromDataFrame(iserverpd, columns=accumulateFeatures)
        # 对特征值进行特征提取
        featureExtractionDf = featureExtractionPd(subtractpd, extraFeature=extractFeatures, windowSize=windowsSize)
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
"""


def getprocess_cputime_abcores(processpds: pd.DataFrame, nowtime: str, isThreshold: bool = False,
                               thresholdValue: Dict = None, modelfilepath: str = None) -> Union[
    tuple[None, None], tuple[Any, list[Union[list, Any]]]]:
    nowdf = processpds[TIME_COLUMN_NAME == nowtime]
    if len(nowdf) == 0:
        return None, None

    # 先得到总的CPUTIME的时间
    cputime = nowdf["cpu"].sum()
    # 核的编号
    cores_serialnumber = list(nowdf.loc[:, CPU_FEATURE])
    if isThreshold:
        predictflag = isCPUAbnormalsByThreshold(nowdf, thresholdValue)
    else:
        predictflag = select_and_pred(nowdf, MODEL_TYPE[0], saved_model_path=modelfilepath)
        predictflag = [True if i != 0 else False for i in predictflag]
    # predictflag为True代表异常， 否则代表这正常
    # 获得异常的核
    assert len(predictflag) == len(cores_serialnumber)
    abnormalcores = [cores_serialnumber[i] for i, flag in range(predictflag) if flag]
    # 将所有的cputime和不正常的核心数据进行返回
    return cputime, abnormalcores


"""
将server文件和process结合，根据时间对数据进行分析，最后得到一个Dict，包含如下信息
time, server_flag(可选), used, used_mean, pgfree, pgfree_mean, pgfree_min, pgfree_max, wrf_cpu, abnormalcore(是一个列表)
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
        wrf_cpu_time, abnormalcores = getprocess_cputime_abcores(allprocesspds, stime, isThreshold=isThreshold,
                                                                 thresholdValue=thresholdValue,
                                                                 modelfilepath=modelfilepath)
        # 不管返回值如何都进行直接的添加
        serverinformationDict["wrf_cpu"].append(wrf_cpu_time)
        serverinformationDict["abnormalcores"] = abnormalcores
    # 将server_flag加入进来, 假如存在的话
    if FAULT_FLAG in allserverpds.columns.array:
        serverinformationDict[FAULT_FLAG] = list(allserverpds[FAULT_FLAG])
    add_server_feature = ["time", "used", "used_mean", "pgfree", "pgfree_mean", "pgfree_amin", "pgfree_amax"]
    for ife in add_server_feature:
        serverinformationDict[ife] = list(allserverpds[ife])

    # 加入一个断言
    assert len(serverinformationDict["wrf_cpu"]) == len(allserverpds)

    # 将字典中的数据进行保存 ==========================================================================================
    nosavefeatures = ["abnormalcores"]
    savedict = dict([(key, serverinformationDict[key]) for key in serverinformationDict.keys() if key not in nosavefeatures])
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
"""


def predictcpu(serverinformationDict: Dict, coresnumber: int = 0) -> List[int]:
    #  wrfnumList不为None
    wrfnumList = serverinformationDict['abnormalcore']
    assert len(serverinformationDict[TIME_COLUMN_NAME]) == len(wrfnumList)
    iscpu = [0]  # 先添加一个数值，最后返回的时候要去掉
    for i, ilist in range(wrfnumList):
        if i == 0:
            iscpu.append(0)  # 第一个我直接预测为正常
            continue
        if len(ilist) == 0:
            iscpu.append(0)
        elif len(ilist) == coresnumber:
            iscpu.append(10)
        elif len(ilist) == 1:
            iscpu.append(20)
        elif len(ilist) == len(wrfnumList[i - 1]) or len(wrfnumList[i - 1]) == 0:
            iscpu.append(30)
        else:
            # cpu数量不等于0，并且前一个不等于，并且和前面的数量不相等
            iscpu[-1] = 80 # 将前一个也改为随机，毕竟前面的就不是多CPU抢占了
            iscpu.append(80)
    return iscpu[1:]

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
        savedict = dict([(key, serverinformationDict[key]) for key in serverinformationDict.keys() if key in used_features])
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
        prelistflag = [60 if i > memorybandwidthValue else 0 for i in realmemorywidthValue]
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
"""
def get_realpredict(predictDict: Dict) -> List:
    cpu_list = predictDict["CPU_Abnormal"]
    leak_list = predictDict["mem_leak"]
    bandwidth_list = predictDict["mem_bandwidth"]

    preflag = []
    for i in range(0, len(cpu_list)):
        if bandwidth_list[i] != 0:
            preflag.append(50)
            continue
        if leak_list[i] != 0:
            preflag.append(60)
            continue
        preflag.append(cpu_list[i])
    return preflag


"""
将得到的基本信息都得到之后，对结果进行分析
如果isThreshold = True 那么就使用阀值预测，否则就使用模型预测，这个时候，模型路径不能为空
coresnumber设置一下，主要是预测全CPU抢占
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
    # 根据CPU信息和得到真是标签值
    predictDict["preFlag"] = get_realpredict(predictDict)

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


if __name__ == "__main__":
    # ============================================================================================= 输入数据定义
    # 先将所有的server文件和process文件进行指定
    # 其中单个server文件我默认是连续的
    predictdirpath = R"C:\Users\lWX1084330\Desktop\正常和异常数据\E5-3km-异常数据"
    predictserverfiles = getfilespath(os.path.join(predictdirpath, "server"))
    predictprocessfiles = getfilespath(os.path.join(predictdirpath, "process"))
    # 指定正常server和process文件路径
    normaldirpath = R"C:\Users\lWX1084330\Desktop\正常和异常数据\E5-3km-正常数据"
    normalserverfiles = getfilespath(os.path.join(normaldirpath, "server"))
    normalprocessfiles = getfilespath(os.path.join(normaldirpath, "process"))
    # 预测CPU的模型路径
    processcpu_modelpath = ""
    # 预测内存泄露的模型路径
    servermemory_modelpath = ""
    # 预测内存带宽的模型路径
    serverbandwidth_modelpath = ""
    # 将一些需要保存的临时信息进行保存路径
    spath = "tmp/总过程分析E5"
    # 是否有存在faultFlag
    isExistFaultFlag = True
    # 核心数据
    coresnumber = 104

    # 需要对server数据进行处理的指标
    server_feature = ["used", "pgfree"]
    # 需要对process数据进行处理的指标
    process_feature = ["user", "system"]

    # 在处理时间格式的时候使用，都被转化为'%Y-%m-%d %H:%M:00' 在这里默认所有的进程数据是同一种时间格式，
    server_time_format = '%Y/%m/%d %H:%M'
    process_time_format = '%Y/%m/%d %H:%M'

    # 预测是否使用阀值, True为使用阀值预测 必须指定thresholdValueDict， False代表使用模型进行预测, 必须设置好模型的路径
    isThreshold = True
    thresholdValueDict = {
        "process_cpu_mean": 57,
        "used": 110, # 不要改key值
        "pgfree": 110
    }

    # ============================================================================================= 先将正常数据和预测数据的指标从磁盘中加载到内存中
    print("将数据从文件中读取".center(40, "*"))
    normalprocesspds = []
    normalserverpds = []
    predictprocesspds = []
    predictserverpds = []

    # 加入time faultFlag特征值
    time_server_feature = server_feature.copy()
    time_process_feature = process_feature.copy()
    # 加入时间是为了process和server的对应， 加入pid 是为了进行分类。加入CPU是为了预测哪个CPU出现了异常
    time_server_feature.extend([TIME_COLUMN_NAME])
    time_process_feature.extend([TIME_COLUMN_NAME, PID_FEATURE, CPU_FEATURE])
    # flagFault要视情况而定
    if isExistFaultFlag:
        time_server_feature.append(FAULT_FLAG)
        time_process_feature.append(FAULT_FLAG)

    # 正常进程数据
    for ifile in normalprocessfiles:
        tpd = getfilepd(ifile)
        tpd = tpd.loc[:, time_process_feature]
        normalprocesspds.append(tpd)
    # 正常服务数据
    for ifile in normalserverfiles:
        tpd = getfilepd(ifile)
        tpd = tpd.loc[:, time_server_feature]
        normalserverpds.append(tpd)
    # 预测进程数据
    for ifile in predictprocessfiles:
        tpd = getfilepd(ifile)
        tpd = tpd.loc[:, time_process_feature]
        predictprocesspds.append(tpd)
    # 预测服务数据
    for ifile in predictserverfiles:
        tpd = getfilepd(ifile)
        tpd = tpd.loc[:, time_server_feature]
        predictserverpds.append(tpd)

    # ============================================================================================= 先对正常数据的各个指标求平均值
    print("先对正常数据的各个指标求平均值".center(40, "*"))
    allnormalserverpd, _ = mergeDataFrames(normalserverpds)
    allnormalprocesspd, _ = mergeDataFrames(normalprocesspds)
    # 得到正常数据的平均值
    normalserver_meanvalue = getDFmean(allnormalserverpd, server_feature)
    normalprocess_meanvalue = getDFmean(allnormalprocesspd, process_feature)
    # 将这几个平均值进行保存
    tpath = os.path.join(spath, "1. 正常数据的平均值")
    if not os.path.exists(tpath):
        os.makedirs(tpath)
    normalprocess_meanvalue.to_csv(os.path.join(tpath, "meanvalue_process.csv"))
    normalserver_meanvalue.to_csv(os.path.join(tpath, "meanvalue_server.csv"))

    # ============================================================================================= 对要预测的数据进行标准化处理
    # 标准化process 和 server数据
    print("标准化要预测的process和server数据".center(40, "*"))
    standard_server_pds = standardLists(pds=predictserverpds, standardFeatures=server_feature,
                                        meanValue=normalserver_meanvalue, standardValue=100)
    standard_process_pds = standardLists(pds=predictprocesspds, standardFeatures=process_feature,
                                         meanValue=normalprocess_meanvalue, standardValue=60)
    standard_process_pds = add_cpu_column(standard_process_pds)  # 针对process数据计算cpu的值
    # ============================================================================================= 对process数据和server数据进行秒数的处理，将秒数去掉
    standard_server_pds = changeTimeTo_pdlists(standard_server_pds, server_time_format)
    standard_process_pds = changeTimeTo_pdlists(standard_process_pds, process_time_format)
    # ============================================================================================= 对process数据和server数据进行特征提取
    print("对process数据进行特征处理".center(40, "*"))
    tpath = os.path.join(spath, "2. process特征提取数据")
    # 将cpu特征添加到process_feature中
    process_feature.append('cpu')
    extraction_process_pds = processpdsList(standard_process_pds, extractFeatures=process_feature,
                                            accumulateFeatures=process_feature, windowsSize=3, spath=tpath)
    print("对server数据进行特征处理".center(40, "*"))
    tpath = os.path.join(spath, "3. server特征提取数据")
    extraction_server_pds = serverpdsList(standard_server_pds, extractFeatures=server_feature,
                                          accumulateFeatures=server_feature, windowsSize=3, spath=tpath)

    # ============================================================================================= 将process数据和server数据合在一起，按照server时间进行预测
    print("将提取之后的server数据和process数据进行合并".center(40, "*"))
    tpath = os.path.join(spath, "4. server和process合并")
    allserverpds, _ = mergeDataFrames(extraction_server_pds)
    allprocesspds, _ = mergeDataFrames(extraction_process_pds)
    serverinformationDict = deal_serverpds_and_processpds(
        allserverpds=allserverpds,
        allprocesspds=allprocesspds,
        spath=tpath,
        isThreshold=isThreshold,
        thresholdValue=thresholdValueDict,
        modelfilepath=processcpu_modelpath
    )
    # ============================================================================================= 对process数据和server数据合在一起进行预测
    print("对server数据和process数据进行预测".center(40, "*"))
    tpath = os.path.join(spath, "5. 最终预测结果")
    predictAllAbnormal(
        serverinformationDict=serverinformationDict,
        spath=tpath,
        isThreshold=isThreshold,
        thresholdValue=thresholdValueDict,
        Memory_bandwidth_modelpath=serverbandwidth_modelpath,
        Memory_leaks_modelpath=servermemory_modelpath,
        coresnumber=coresnumber
    )
