import itertools
import json
import os.path
import re
from collections import defaultdict
from typing import List, Dict, Union, Any, Set, Tuple, Optional

import numpy as np
import pandas as pd

from Classifiers.ModelPred import select_and_pred, select_and_pred_probability
from utils.DataFrameOperation import mergeDataFrames, SortLabels, PushLabelToFirst, PushLabelToEnd, \
    subtractLastLineFromDataFrame
from utils.DataScripts import TranslateTimeListStrToStr, standardPDfromOriginal1, removeTimeAndfaultFlagFromList
from utils.DefineData import TIME_COLUMN_NAME, FAULT_FLAG, PID_FEATURE, CPU_FEATURE, MODEL_TYPE, CPU_ABNORMAL_TYPE, \
    MEMORY_ABNORMAL_TYPE


def getfilepd(ipath: str, features: List[str] = None) -> pd.DataFrame:
    if not os.path.exists(ipath):
        filename = os.path.basename(ipath)
        print("{} 文件不存在".format(filename))
        exit(1)
    tpd = pd.read_csv(ipath)
    if features is not None:
        return tpd[:, features]
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
# 会修改原数据
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
根据字符串得到时间串格式
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
将时间都转化为标准格式
处理完之后的格式为这种形式
'%Y-%m-%d %H:%M:00'
leastTime 精确到那个时间点， 如果精确到分钟，就将秒进行修改
"""


# 将一个pd中的时间序列的秒变为0
def changeTimeColumns(df: pd.DataFrame, leastTime: str = "%M", timefeaturename: str=TIME_COLUMN_NAME) -> pd.DataFrame:
    if len(df) == 0:
        return df
    stimestr = df[timefeaturename][0]
    # timeformat给出的选项仅仅供选择，下面进行自动生成格式选项
    timeformat = getTimeFormat(stimestr)
    tpd = df.loc[:, [timefeaturename]].apply(lambda x: TranslateTimeListStrToStr(x.to_list(), timeformat, leastTime=leastTime), axis=0)
    df.loc[:, timefeaturename] = tpd.loc[:, timefeaturename]
    return df


# 讲一个pd的列表全都改变 timeformat没有使用的作用
def changeTimeTo_pdlists(pds: List[pd.DataFrame], timeformat: str = '%Y/%m/%d %H:%M', leastTime: str="%M", timefeaturename: str=TIME_COLUMN_NAME) -> List[pd.DataFrame]:
    changed_pds = []
    for ipd in pds:
        tpd = changeTimeColumns(ipd, leastTime=leastTime, timefeaturename=TIME_COLUMN_NAME)
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

        quantile_.__name__ = 'percentage%d' % (n * 100)
        return quantile_

    # 得到的这个DataFrame是一个二级列名类似下面这种
    #	    system	                            ｜                      user
    #  sum	mean	amax	amin	quantile50  ｜ sum	mean	amax	amin	quantile50
    # 由于max和min的函数名字是amax和amin 所以需要修改，记得还原，防止引起不必要的麻烦
    maxname = np.max.__name__
    minname = np.min.__name__
    np.max.__name__ = "max"
    np.min.__name__ = "min"
    featureExtractionDf = df.loc[:, extraFeature].rolling(window=windowSize, min_periods=1, center=True).agg(
        [np.mean, np.max, np.min, quantile(0.5)])
    np.max.__name__ = maxname
    np.min.__name__ = minname

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
        assert len(idf) > 6
        idf = idf.iloc[3:-3]  ## 删除数据了
        print("size: {}".format(len(idf)))
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
返回值： 1. 异常的数量列表  2. 异常的核心列表 3. 异常的cpu最小值列表 4. 所有核心上的预测为10的概率是一个core-概率的字典结构
"""


def getprocess_cputime_abcores(processpds: pd.DataFrame, nowtime: str, isThreshold: bool = False,
                               thresholdValue: Dict = None, modelfilepath: str = None, modeltype=0) -> Union[
    tuple[int, None, None, None], tuple[Any, list, list, Optional[Any]]]:
    nowdf = processpds[processpds[TIME_COLUMN_NAME] == nowtime]
    predictflag_probability = None  # 存储概率的列表
    if len(nowdf) == 0:
        return 0, None, None, None

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
        predictflag = select_and_pred(nowdf, MODEL_TYPE[modeltype], saved_model_path=modelfilepath)
        predictflag_probabilityDict = select_and_pred_probability(nowdf, MODEL_TYPE[modeltype],
                                                                  saved_model_path=modelfilepath)
        if 10 in predictflag_probabilityDict.keys():
            predictflag_probability = predictflag_probabilityDict[10]
            predictflag_probability = dict(zip(cores_serialnumber, predictflag_probability))
        predictflag = [True if i != 0 else False for i in predictflag]
    # predictflag为True代表异常， 否则代表这正常
    # 获得异常的核
    assert len(predictflag) == len(cores_serialnumber)
    abnormalcores = [cores_serialnumber[i] for i, flag in enumerate(predictflag) if flag]
    abnormalcoremaxtime = [cores_max_time[i] for i, flag in enumerate(predictflag) if flag]
    # 将所有的cputime和不正常的核心数据进行返回
    return cputime, abnormalcores, abnormalcoremaxtime, predictflag_probability


"""

输入：addserverfeatures 包含所需要添加的server特征值 可能包含time 和 faultFlag， 最后返回的dict中将会包含这写特征值的扩展特征值

将server文件和process结合，根据时间对数据进行分析，最后得到一个Dict，包含如下信息
time, server_flag(可选), used, used_mean, pgfree, pgfree_mean, pgfree_min, pgfree_max, wrf_cpu, abnormalcore(是一个列表)
返回值key 
time    wrf_cpu    abnormalcores    faultyFlag   "used", "used_mean", "pgfree", "pgfree_mean", "pgfree_amin", "pgfree_amax"  coresnums
serverinformationDict: 
abnormalcores 是一个列表的列表，存储的是异常的核心数
coresnums 是指多少个核心出现了异常
coresmaxtime: 是一个列表的列表， 存储的是每个异常核心的值
coresallprobability: 是每个时刻中预测每个核为异常的概率
"""


def deal_serverpds_and_processpds(allserverpds: pd.DataFrame, allprocesspds: pd.DataFrame, spath: str = None,
                                  isThreshold: bool = False, thresholdValue: Dict = None,
                                  modelfilepath: str = None,
                                  modeltype=0,
                                  addserverfeatures=None) -> Dict:
    if addserverfeatures is None:
        addserverfeatures = ["used", "pgfree"]
    if TIME_COLUMN_NAME in addserverfeatures:
        addserverfeatures.remove(TIME_COLUMN_NAME)
    if FAULT_FLAG in addserverfeatures:
        addserverfeatures.remove(FAULT_FLAG)

    # 往dict中添加所需要的特征值
    add_server_feature = [TIME_COLUMN_NAME]
    for i in addserverfeatures:
        add_server_feature.append(i)
        add_server_feature.append("{}_mean".format(i))
        add_server_feature.append("{}_min".format(i))
        add_server_feature.append("{}_max".format(i))
        add_server_feature.append("{}_percentage50".format(i))

    if spath is not None and not os.path.exists(spath):
        os.makedirs(spath)
    # 将allserverpds里面所有的时间搜集起来
    timecolumns = allserverpds[TIME_COLUMN_NAME]
    serverinformationDict = defaultdict(list)
    for stime in timecolumns:
        # 添加wrf的cpu时间
        wrf_cpu_time, abnormalcores, abnormalcoremaxtime, predictflag_probability = getprocess_cputime_abcores(
            allprocesspds, stime,
            isThreshold=isThreshold,
            thresholdValue=thresholdValue,
            modelfilepath=modelfilepath,
            modeltype=modeltype)
        # 不管返回值如何都进行直接的添加
        serverinformationDict["wrf_cpu"].append(wrf_cpu_time)
        serverinformationDict["abnormalcores"].append(abnormalcores)
        serverinformationDict["coresmaxtime"].append(abnormalcoremaxtime)
        serverinformationDict["coresallprobability"].append(predictflag_probability)  # 所有核心上的概率
    # 将server_flag加入进来, 假如存在的话
    if FAULT_FLAG in allserverpds.columns.array:
        serverinformationDict[FAULT_FLAG] = list(allserverpds[FAULT_FLAG])
    # add_server_feature = ["time", "used", "used_mean", "used_min", "used_max", "used_percentage50", "pgfree", "pgfree_mean", "pgfree_min", "pgfree_max", "pgfree_percentage50"]
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
serverinformationDict: 
abnormalcores 是一个列表的列表，存储的是异常的核心数
coresnums 是指多少个核心出现了异常
coresmaxtime: 是一个列表的列表， 存储的是每个异常核心的值
coresallprobability: 是每个时刻中预测每个核为异常的概率, 是一个core-概率的字典结构
"""


def getCPU_Abnormal_Probability(serverinformationDict: Dict, coresnumber: int = 0) -> List[int]:
    coresallprobabilityList = serverinformationDict["coresallprobability"]
    abnormalcoresList = serverinformationDict["abnormalcores"]  # 如果这里出现了None， 那么表明这个时刻不存在
    assert len(abnormalcoresList) == len(coresallprobabilityList)

    # 根据核心获得核心的概率函数
    def getProbability(abnormalcores: List[int], coresallprobability: Dict) -> List:
        resList = []
        for icore in abnormalcores:
            resList.append(coresallprobability[icore])
        return resList

    minprobabilityList = []
    for i in range(0, len(abnormalcoresList)):
        tprobabilityDict = coresallprobabilityList[i]
        tabnormalcores = abnormalcoresList[i]
        if tabnormalcores is None:
            minprobabilityList.append(-1)
            continue
        if len(tabnormalcores) == 0:  # 正常情况
            minprobabilityList.append(min(tprobabilityDict.values()))
            continue
        minp = min(getProbability(abnormalcores=tabnormalcores, coresallprobability=tprobabilityDict))
        minprobabilityList.append(minp)
    return minprobabilityList


"""
对内存泄露进行预测
有两种方式，一种是通过阀值，一种是通过模型， 如果是通过模型则需要生成一个dataFrame传递进去
"""


def predict_memory_leaks(serverinformationDict: Dict, isThreshold: bool = False, thresholdValue: Dict = None,
                         Memory_leaks_modelpath: str = None, mem_leak_features=None, memory_leaks_modeltype=0) -> tuple[
    Union[list[int], Any], Optional[Any]]:
    if mem_leak_features is None:
        mem_leak_features = ["used"]
    if TIME_COLUMN_NAME in mem_leak_features:
        mem_leak_features.remove(TIME_COLUMN_NAME)
    if FAULT_FLAG in mem_leak_features:
        mem_leak_features.remove(FAULT_FLAG)

    prelistflag_probability = None  # 使用阈值的时候预测的概率只能为None
    if isThreshold:
        memoryleakValue = thresholdValue["used"]
        realmemoryleakValue = serverinformationDict["used_mean"]
        prelistflag = [60 if i > memoryleakValue else 0 for i in realmemoryleakValue]
    else:
        # 先构造一个字典，然后生成dataFrame, 调用接口进行预测
        used_features = []  # 得到预测内存泄露的特征值
        for i in mem_leak_features:
            used_features.append("{}".format(i))
            used_features.append("{}_mean".format(i))
            used_features.append("{}_min".format(i))
            used_features.append("{}_max".format(i))
            used_features.append("{}_percentage50".format(i))

        savedict = dict(
            [(key, serverinformationDict[key]) for key in serverinformationDict.keys() if key in used_features])
        tpd = pd.DataFrame(data=savedict)
        # 得到预测值
        prelistflag = select_and_pred(tpd, MODEL_TYPE[memory_leaks_modeltype], saved_model_path=Memory_leaks_modelpath)
        # 得到预测的概率
        prelistflag_probabilityDict = select_and_pred_probability(tpd, MODEL_TYPE[memory_leaks_modeltype],
                                                                  saved_model_path=Memory_leaks_modelpath)
        if 60 in prelistflag_probabilityDict.keys():
            prelistflag_probability = prelistflag_probabilityDict[60]
    return prelistflag, prelistflag_probability


"""
对内存带宽进行预测
有两种方式，一种是通过阀值，一种是通过模型， 如果是通过模型则需要生成一个dataFrame传递进去

mem_bandwidth_features: 预测内存带宽所需要的特征值
"""


def predict_memory_bandwidth(serverinformationDict: Dict, isThreshold: bool = False, thresholdValue: Dict = None,
                             Memory_bandwidth_modelpath: str = None, mem_bandwidth_features=None,
                             memory_bandwidth_modeltype=0) -> tuple[list[int], Optional[Any]]:
    if mem_bandwidth_features is None:
        mem_bandwidth_features = ["pgfree"]
    if TIME_COLUMN_NAME in mem_bandwidth_features:
        mem_bandwidth_features.remove(TIME_COLUMN_NAME)
    if FAULT_FLAG in mem_bandwidth_features:
        mem_bandwidth_features.remove(FAULT_FLAG)

    prelistflag_probability = None  # 使用阈值的时候预测的概率只能为None
    if isThreshold:
        memorybandwidthValue = thresholdValue["pgfree"]
        realmemorywidthValue = serverinformationDict["pgfree_mean"]
        prelistflag = [50 if i > memorybandwidthValue else 0 for i in realmemorywidthValue]
    else:
        # 先构造一个字典，然后生成dataFrame, 调用接口进行预测
        # used_features = ["pgfree_mean", "pgfree_max", "pgfree_min", "pgfree_percentage50"]
        used_features = []  # 得到预测内存泄露的特征值
        for i in mem_bandwidth_features:
            used_features.append("{}".format(i))
            used_features.append("{}_mean".format(i))
            used_features.append("{}_min".format(i))
            used_features.append("{}_max".format(i))
            used_features.append("{}_percentage50".format(i))
        savedict = dict(
            [(key, serverinformationDict[key]) for key in serverinformationDict.keys() if key in used_features])
        tpd = pd.DataFrame(data=savedict)
        prelistflag = select_and_pred(tpd, MODEL_TYPE[memory_bandwidth_modeltype],
                                      saved_model_path=Memory_bandwidth_modelpath)
        # 得到预测的概率
        prelistflag_probabilityDict = select_and_pred_probability(tpd, MODEL_TYPE[memory_bandwidth_modeltype],
                                                                  saved_model_path=Memory_bandwidth_modelpath)
        if 50 in prelistflag_probabilityDict.keys():
            prelistflag_probability = prelistflag_probabilityDict[50]
    prelistflag = [50 if i == 50 else 0 for i in prelistflag]
    return prelistflag, prelistflag_probability


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
        if preflag[i - 2] != 0 and preflag[i - 1] != 0 and preflag[i] == 0 and preflag[i + 1] != 0 and preflag[
            i + 2] != 0:
            preflag[i] = preflag[i - 1]  # 如果遇到这种情况，取得上一个数
        if preflag[i - 2] == 0 and preflag[i - 1] == 0 and preflag[i] != 0 and preflag[i + 1] == 0 and preflag[
            i + 2] == 0:
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

输入参数：
mem_leak_features 是内存泄露所需要的特征，这里面的特征是used这些原始指标，会自动进行扩充
mem_bandwidth_features 类似
memory_bandwidth_modeltype 0-决策树  1-随机森林  2-自适应增强


"""


def predictAllAbnormal(serverinformationDict: Dict, spath: str, isThreshold: bool = False,
                       thresholdValue: Dict = None,
                       Memory_bandwidth_modelpath: str = None, Memory_leaks_modelpath: str = None,
                       memory_bandwidth_modeltype=0, memory_leaks_modeltype=0,
                       coresnumber: int = 0,
                       mem_leak_features=None,
                       mem_bandwidth_features=None,
                       ) -> pd.DataFrame:
    if mem_bandwidth_features is None:
        mem_bandwidth_features = ["pgfree"]
    if mem_leak_features is None:
        mem_leak_features = ["used"]
    predictDict = {}
    predictDict[TIME_COLUMN_NAME] = serverinformationDict[TIME_COLUMN_NAME]
    if FAULT_FLAG in serverinformationDict.keys():
        predictDict[FAULT_FLAG] = serverinformationDict[FAULT_FLAG]
    # 对CPU进行预测
    predictDict["CPU_Abnormal"] = predictcpu(serverinformationDict, coresnumber)
    # 对内存泄露进行预测
    predictDict["mem_leak"], predict_probability = predict_memory_leaks(
        serverinformationDict=serverinformationDict,
        isThreshold=isThreshold,
        thresholdValue=thresholdValue,
        Memory_leaks_modelpath=Memory_leaks_modelpath,
        mem_leak_features=mem_leak_features,
        memory_leaks_modeltype=memory_leaks_modeltype
    )
    # 对内存带宽进行预测
    predictDict["mem_bandwidth"], predict_probability = predict_memory_bandwidth(
        serverinformationDict=serverinformationDict,
        isThreshold=isThreshold,
        thresholdValue=thresholdValue,
        Memory_bandwidth_modelpath=Memory_bandwidth_modelpath,
        mem_bandwidth_features=mem_bandwidth_features,
        memory_bandwidth_modeltype=memory_bandwidth_modeltype,
    )
    # 得到核的数量
    predictDict["coresnums"] = serverinformationDict["coresnums"]
    # 根据CPU信息和得到真是标签值
    predictDict["preFlag"] = get_realpredict(predictDict, serverinformationDict["abnormalcores"])
    predictDict["概率"] = getProbability(list(predictDict["preFlag"]))
    # 某一时刻的cpu列表
    wrfnumList = serverinformationDict['abnormalcores']
    # 得到某一时刻下
    predictDict["smaxcputime"] = getSingleMaxCPUTime(serverinformationDict)
    predictDict["pgfree_mean"] = serverinformationDict["pgfree_mean"]
    predictDict["used_mean"] = serverinformationDict["used_mean"]

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
    if len(accumulateFeatures) == 0:
        return processpds
    differencepds = []
    for iprocesspd in processpds:
        subtractpdLists = []
        for ipid, ipd in iprocesspd.groupby(PID_FEATURE):
            # 先将一些不可用的数据进行清除,比如一个进程只运行了两分钟
            if len(ipd) <= 6:
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
    if len(accumulateFeatures) == 0:
        return serverpds
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
    allpd, _ = mergeDataFrames(removepds)
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
    assert len(realflag) == len(preflag)  # 断言
    # 得到realflag异常的时间段
    tlist1 = [False if i == 0 else True for i in realflag]
    tlist = [k for k, g in itertools.groupby(tlist1)]  # 去除重复
    realperiodLen = len([i for i in tlist if i])  # 找到异常的数值

    # 得到preflag异常的时间段
    tlist2 = [False if i == 0 else True for i in preflag]
    tlist = [k for k, g in itertools.groupby(tlist)]
    preperiodLen = len([i for i in tlist if i])

    #
    sameLen = 0
    for i in range(0, len(tlist) - 3):
        if tlist1[i] and tlist1[i + 1] and tlist1[i + 2] and tlist2[i] and tlist2[i + 1] and tlist2[i + 2]:
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
    realflags = list(predictpd[FAULT_FLAG])  # realflags可能是11 12 13这种代表各个强度的数值
    preflags = list(predictpd[preflaglabel])  # preflags都是整数，比如10，20，30
    assert len(realflags) == len(preflags)
    rightflagSet = set([(i // 10) * 10 for i in abnormalsSet])  # 如果预测在这个集合中， 则认为预测正确

    real_abnormalnums = 0  # 异常的总数量
    pre_allabnormalnums = 0  # 所有预测数据中，被预测为异常的数量
    abnormal_rightabnormal_nums = 0  # 异常被预测为正确的个数
    abnormal_abnormal_nums = 0  # 异常被预测为!=0的数量
    abnormal_normal_nums = 0  # 异常被预测为正常的数量
    abnormal_memory_nums = 0  # 异常被预测为内存异常的数量
    abnormal_cpu_nums = 0  # 异常被预测为cpu异常的数量

    for i in range(len(realflags)):
        if realflags[i] in abnormalsSet:
            real_abnormalnums += 1  # 表示异常的真实数量+1
        if preflags[i] in rightflagSet:
            pre_allabnormalnums += 1  # 被预测为异常的真实数量+1

        if realflags[i] in abnormalsSet:
            # 现在实际预测值是异常
            if preflags[i] == 0:
                # 预测值是0
                abnormal_normal_nums += 1
            if preflags[i] != 0:
                abnormal_abnormal_nums += 1
            if preflags[i] in rightflagSet:
                abnormal_rightabnormal_nums += 1  # 异常预测正确
            if preflags[i] in CPU_ABNORMAL_TYPE:
                abnormal_cpu_nums += 1
            if preflags[i] in MEMORY_ABNORMAL_TYPE:
                abnormal_memory_nums += 1

    infoDict["num"] = real_abnormalnums
    infoDict["recall"] = -1 if real_abnormalnums == 0 else abnormal_rightabnormal_nums / real_abnormalnums
    infoDict["precison"] = -1 if pre_allabnormalnums == 0 else abnormal_rightabnormal_nums / pre_allabnormalnums
    infoDict[
        "per_abnormal"] = -1 if real_abnormalnums == 0 else abnormal_abnormal_nums / real_abnormalnums  # 预测为异常的比例, 异常的发现率
    infoDict["per_normal"] = -1 if real_abnormalnums == 0 else abnormal_normal_nums / real_abnormalnums  # 预测为正常的比例
    infoDict["cpu_abnormal"] = -1 if real_abnormalnums == 0 else abnormal_cpu_nums / real_abnormalnums
    infoDict["memory_abnormal"] = -1 if real_abnormalnums == 0 else abnormal_memory_nums / real_abnormalnums
    infoDict["f-score"] = harmonic_mean([infoDict["recall"], infoDict["precison"]])
    return infoDict


"""
输入：一个是实际标签值  一个是预测标签值
实际标签值是一个包含各种强度11,12,13的int列表
预测标签纸是不包含强度信息的int列表
excludeflags 要排除预测的标签值 是指 51 52这种数值  不是50数值
输出： 一个小数 表示预测的准确率
"""


def getAccuracy(realflags: List[int], preflags: List[int], excludeflags=None) -> float:
    if excludeflags is None:
        excludeflags = []
    assert len(realflags) == len(preflags)
    # 得到预测对的数量
    # rightnumber = len([i for i in range(0, len(realflags)) if (realflags[i] // 10) * 10 == preflags[i]])
    allnumber = 0
    rightnumber = 0
    for i in range(0, len(realflags)):
        if realflags[i] in excludeflags:
            continue
        allnumber += 1
        if (realflags[i] // 10) * 10 == preflags[i]:
            rightnumber += 1
    return rightnumber / allnumber


# time  faultFlag  preFlag  mem_leak  mem_bandwidth
# 主要分析三种情况，1. 不去除首位的，2. 去除首位  3. 去除低等级
# 得到10 20 30 50 60 以及 将10 20 30当作cpu 一种情况
def analysePredictResult(predictpd: pd.DataFrame, spath: str, windowsize: int = 3):
    predictorginedpd = predictpd.copy()
    # 先将{40, 70, 90} 这三种异常去除,并且去除其首尾数据
    predictpd = remove_Abnormal_Head_Tail(predictpd, windowsize=windowsize, abnormals={
        41, 42, 43, 44, 45,
        71, 72, 73, 74, 75,
        91, 92, 93, 94, 95
    })
    preflaglabel = "preFlag"
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
    accuracy1 = getAccuracy(list(predictpd[FAULT_FLAG]), list(predictpd[preflaglabel]))
    accuracy1_nonormal = getAccuracy(list(predictpd[FAULT_FLAG]), list(predictpd[preflaglabel]), excludeflags=[0])

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
    accuracy2 = getAccuracy(list(tpd[FAULT_FLAG]), list(tpd[preflaglabel]))
    accuracy2_nonormal = getAccuracy(list(tpd[FAULT_FLAG]), list(tpd[preflaglabel]), excludeflags=[0])
    # ===============================================================================统计时间段信息
    realperiodLen, preperiodLen, sameLen = getTimePeriodInfo(tpd)
    writeinfo = ["实际时间段个数:{}\n".format(realperiodLen), "预测时间段个数:{}\n".format(preperiodLen), "预测准确时间段个数:{}\n". \
        format(sameLen), "预测召回率:{:.2%}\n".format(sameLen / realperiodLen),
                 "预测精确率:{:.2%}\n".format(sameLen / preperiodLen)]
    # ===============================================================================
    # 将信息进行保存
    tpd = pd.DataFrame(data=analyseDict).T
    tpd.to_csv(os.path.join(spath, "2. 去除首位_统计数据.csv"))

    with open(os.path.join(spath, "2. 去除首位_统计信息时间段.txt"), "w", encoding="utf-8") as f:
        f.writelines(writeinfo)
    # =============================================================================== 得到去除首尾之后的准确率

    # 预测去除低等级之后的数据 包括首尾数据===============================================================================
    tpd = remove_Abnormal_Head_Tail(predictPd=predictpd, windowsize=windowsize, abnormals={
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
    accuracy3 = getAccuracy(list(tpd[FAULT_FLAG]), list(tpd[preflaglabel]))
    accuracy3_nonormal = getAccuracy(list(tpd[FAULT_FLAG]), list(tpd[preflaglabel]), excludeflags=[0])
    # 将信息进行保存
    tpd = pd.DataFrame(data=analyseDict).T
    tpd.to_csv(os.path.join(spath, "3. 去除首位_去除低强度_统计数据.csv"))
    # ================================================================================================== 统计全文准确率
    # 将三种情况得到的准确率写入
    writeinfo = [
        "不去除首尾准确率:{:.2%}\n".format(accuracy1),
        "去除首尾准确率:{:.2%}\n".format(accuracy2),
        "去除首尾_去除低强度:{:.2%}\n".format(accuracy3),
        "=================================去除正常情况统计准确率\n",
        "不去除首尾准确率:{:.2%}\n".format(accuracy1_nonormal),
        "去除首尾准确率:{:.2%}\n".format(accuracy2_nonormal),
        "去除首尾_去除低强度:{:.2%}\n".format(accuracy3_nonormal),
    ]
    with open(os.path.join(spath, "4. 准确率.txt"), "w", encoding="utf-8") as f:
        f.writelines(writeinfo)
    # ================================================================================================= 对时间段的输出
    # 传入进去的是一个predictpd的DataFrame 其中要包含FAULT_FLAG和preflaglabel
    timeperiodPd = getDetailedInformationOnTime(predictorginedpd)
    # 进行保存
    timeperiodPd: pd.DataFrame
    timeperiodPd.to_csv(os.path.join(spath, "5. 不去除首尾-详细时间段信息.csv"))

    # 演示效果
    time7path = os.path.join(os.path.dirname(spath), "7.关键信息输出")
    outputRestult(rpath=spath, spath=time7path)


"""
输入：包含一个实际值和预测值的DataFrame
输出：预测的时间段信息 以DataFrame形式输出
"""


def getDetailedInformationOnTime(predictpd: pd.DataFrame, timeLabelName: str = "time", realFlagName: str = "faultFlag",
                                 testFlagName: str = "preFlag", probabilityName: str = "概率") -> pd.DataFrame:
    # ========================================================================================================== 函数部分
    # 找到一个列表中连续不为0的数据的位置, 返回的是每段不为0的起始位置[4,9), 左闭右开
    def findAbnormalPos(flags: List[int]) -> List[Tuple[int, int]]:
        beginpos_endpos_List = []
        i = 0
        while i < len(flags):
            if flags[i] == 0:
                i += 1
                continue
            beginpos = i
            while i < len(flags) and flags[i] != 0:
                i += 1
            endpos = i
            beginpos_endpos_List.append((beginpos, endpos))
        return beginpos_endpos_List

    # 根据位置的起始位置得到DataFrame
    def getDataFramesFromPos(pd: pd.DataFrame, pos: List[Tuple[int, int]]) -> List[pd.DataFrame]:
        pd = pd.copy()
        respdList = []
        for i in pos:
            # loc和使用[]是不一样的
            respdList.append(pd[i[0]:i[1]])
        return respdList

    # 判断两个DataFrame是否交叉，如果交叉返回True，DataFrame  否则 False，DataFrame
    def determineTwoDataframeOverlap(df1: pd.DataFrame, df2: pd.DataFrame) -> Union[
        tuple[bool, None], tuple[bool, Any]]:
        df1times = set(df1[timeLabelName])
        df2times = set(df2[timeLabelName])
        overlapTime = list(df1times & df2times)
        if len(overlapTime) == 0:
            return False, None

        # 获得时间
        def f1(x):
            if x in overlapTime:
                return True
            return False

        return True, df1.loc[df1[timeLabelName].apply(f1)]

    # 判断一个DataFrame的时间是否与一个时间列表交叉，如果交叉返回交叉的True, DataFrame 否则 False，DataFrame
    # 返回是否交叉  返回交叉的部分  返回匹配到交叉的部分
    def determineDataframeListOverlap(df: pd.DataFrame, dflist: List[pd.DataFrame]) -> Union[
        tuple[bool, Any, Any], tuple[bool, None, None]]:
        for idf in dflist:
            iscross, crossdf = determineTwoDataframeOverlap(df, idf)
            if iscross:
                return True, crossdf, idf
        return False, None, None

    # 得到列表中出现的最大频率的数值，以及去重之后的列表
    def getMaxNumLabels(labels: List):
        prelabels = max(labels, key=labels.count)
        alllabeslList = sorted(list(set(labels)))
        return prelabels, alllabeslList

    """
    得到概率
    参数： iprepd-预测到的时间段  tcrosspd-交叉的时间段 trealpd-实际的时间段
    返回值 0: CPU概率   1: 内存泄露概率   2: 内存带宽异常
    
    """

    def getProbability(iprepd: pd.DataFrame, tcrosspd: pd.DataFrame, trealpd: pd.DataFrame) -> List:
        nowtimeProbability = iprepd[probabilityName].mean()
        return nowtimeProbability

    # ====================================================================================================== 函数部分结束
    # =================================================================================================得到时间段的逻辑部分
    testLabelName = testFlagName
    reallabels = list(predictpd[realFlagName])
    prelabels = list(predictpd[testLabelName])
    # =================================================================================================得到真实标签的分类
    beginpos_endpos_list = findAbnormalPos(reallabels)
    realTimePeriodAbnormalPds = getDataFramesFromPos(predictpd, beginpos_endpos_list)
    # =================================================================================================得到预测标签的分类
    beginpos_endpos_list = findAbnormalPos(prelabels)
    preTimePeriodAbnormalPds = getDataFramesFromPos(predictpd, beginpos_endpos_list)
    # =================================================================================================时间段的逻辑

    timeperiodDict = defaultdict(list)
    for iprepd in preTimePeriodAbnormalPds:
        assert len(iprepd) != 0
        prebegintime = iprepd[timeLabelName].iloc[0]  # 预测开始时间
        timeperiodDict["检测开始时间"].append(prebegintime)
        preendtime = iprepd[timeLabelName].iloc[-1]  # 预测结束时间
        timeperiodDict["检测结束时间"].append(preendtime)
        preLastime = len(iprepd)  # 预测持续时间
        timeperiodDict["检测运行时间"].append(preLastime)
        maxNumLabels, preAllLabels = getMaxNumLabels(list(iprepd[testLabelName]))  # 得到当前预测时间内的预测值
        timeperiodDict["检测标记"].append(maxNumLabels)
        timeperiodDict["检测所有标记"].append(",".join([str(i) for i in preAllLabels]))

        # 判断是否有真实标签值与其重叠
        iscross, tcrosspd, trealpd = determineDataframeListOverlap(iprepd, realTimePeriodAbnormalPds)
        # 得到概率
        nowtimeProbability = getProbability(iprepd=iprepd, tcrosspd=tcrosspd, trealpd=trealpd)
        timeperiodDict["概率"].append(nowtimeProbability)

        realcrossBeginTime = str(-1)
        realcrossEndTime = str(-1)
        crossTime = 0
        realTimeLen = 0
        realcrossLabels = 0
        if iscross:
            assert len(tcrosspd) != 0
            realcrossBeginTime = trealpd[timeLabelName].iloc[0]
            realcrossEndTime = trealpd[timeLabelName].iloc[-1]
            crossTime = len(tcrosspd)
            realTimeLen = len(trealpd)

            realcrossLabels, _ = getMaxNumLabels(list(trealpd[realFlagName]))
        timeperiodDict["实际开始时间"].append(realcrossBeginTime)
        timeperiodDict["实际结束时间"].append(realcrossEndTime)
        timeperiodDict["实际运行时间"].append(realTimeLen)
        timeperiodDict["重叠时间"].append(crossTime)
        timeperiodDict["实际标记"].append(realcrossLabels)

    timeperiodDictPd = pd.DataFrame(data=timeperiodDict)
    return timeperiodDictPd


"""
对server数据列表中pgfree进行滑动窗口的处理
会将传入参数的列表中dataframe本身数值进行修改
"""


def smooth_pgfree(serverpds: List[pd.DataFrame], smoothwinsize: int = 6) -> List[pd.DataFrame]:
    pgfree_name = "pgfree"
    for ipd in serverpds:
        if pgfree_name in ipd.columns.array:
            ipd[pgfree_name] = ipd[pgfree_name].rolling(window=smoothwinsize, min_periods=1, center=True).median()
    return serverpds


# 将dataframe中指定指标进行平滑，默认是使用除了time和faultflag
# 会将serverpds本身的值进行修改
def smooth_dfs(serverpds: List[pd.DataFrame], smoothwinsize: int = 7, features: List[str] = None) -> List[pd.DataFrame]:
    if features is None:
        features = list(serverpds[0].columns.array)
    # 去除特征值中的time和faultFlag
    removeTimeAndfaultFlagFromList(features, inplace=True)
    for ipd in serverpds:
        ipd[features] = ipd[features].rolling(window=smoothwinsize, min_periods=1, center=True).median()


"""
提取一个核心上的各个错误，可以保证传入的DataFrame是一个核心上的数据
会进行首尾数据的去除
返回的是一个，
"""


def allMistakesOnExtractingOneCore(onecorePd: pd.DataFrame, windowsize: int = 2) -> Dict:
    faultPdDict = {}
    # 首先是去掉所有异常的首尾
    ridForeAftPD = removeAllHeadTail(onecorePd, windowsize=windowsize)
    for ifault, ipd in ridForeAftPD.groupby(FAULT_FLAG):
        faultPdDict[ifault] = ipd
    return faultPdDict


"""
将所有核上的数据进行提取
"""


def allMistakesOnExtractingAllCore(processpd: pd.DataFrame, windowsize: int = 2) -> Dict:
    core_faultpdDict = {}
    for icore, ipd in processpd.groupby(CPU_FEATURE):
        faultPdDict = allMistakesOnExtractingOneCore(ipd, windowsize=windowsize)
        core_faultpdDict[icore] = faultPdDict
    return core_faultpdDict


"""
根据预测标签值得到概率
0-> 代表不同  1->代表相同
"""


def getProbability(preFlags: List[int]) -> List[float]:
    def getBeforeData(preFlags: List[int], ipos: int, judgelen: int) -> List[int]:
        res = []
        for bpos in range(ipos - judgelen, ipos, 1):
            if bpos < 0:
                res.append(0)
                continue
            if preFlags[ipos] == preFlags[bpos]:
                res.append(1)
            else:
                res.append(0)
        return res

    def getAfterData(preFlags: List[int], ipos: int, judgelen: int) -> List[int]:
        res = []
        for bpos in range(ipos + 1, ipos + judgelen + 1, 1):
            if bpos >= len(preFlags):
                res.append(0)  # 不同
                continue
            if preFlags[ipos] == preFlags[bpos]:
                res.append(1)
            else:
                res.append(0)
        return res

    """
    传入是否重复
    """

    def getp(isSameFlags: List[int], probabilities: List[float]) -> float:
        assert len(isSameFlags) == len(probabilities)
        return sum(isSameFlags[i] * probabilities[i] for i in range(0, len(isSameFlags)))

    # 得到时间概率的逻辑部分
    probabilityList = []
    preprobability = [0.1, 0.2, 0.3, 0.4]
    reverseprobability = list(reversed(preprobability))  # 概率相反
    judgeLen = len(preprobability)  # 概率的长度
    for ipos, iflag in enumerate(preFlags):
        bD = getBeforeData(preFlags, ipos, judgeLen)
        aD = getAfterData(preFlags, ipos, judgeLen)
        nowprobability = 0.1 + 0.45 * (getp(bD, preprobability)) + 0.45 * (getp(aD, reverseprobability))
        probabilityList.append(nowprobability)
    return probabilityList


# 这个函数主要针对演示使用
def outputRestult(rpath: str, spath: str):
    if not os.path.exists(spath):
        os.makedirs(spath)
    # ==========================================================================================================输出准确率
    filepath = os.path.join(rpath, "4. 准确率.txt")
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    writefilepath = os.path.join(spath, "准确率.txt")
    with open(writefilepath, "w") as f:
        acc = lines[5].split(":")[1]
        print("准确率: {}".format(acc))
        f.write("准确率: {}".format(acc))
    # ==========================================================================================================输出时间段
    filepath = os.path.join(rpath, "5. 不去除首尾-详细时间段信息.csv")
    writefilepath = os.path.join(spath, "持续时间.csv")
    feas = ["检测开始时间", "实际开始时间", "检测结束时间", "实际结束时间", "检测标记", "实际标记", "概率", "检测运行时间", "实际运行时间", "重叠时间"]
    tpd = pd.read_csv(filepath)[feas]
    tpd.to_csv(writefilepath, index=False)

    # ==========================================================================================================输出每个点的结果情况
    filepath = os.path.join(rpath, "预测结果.csv")
    writefilepath = os.path.join(spath, "统计数据.csv")
    acfeas = ["time", "faultFlag", "preFlag", "概率"]
    prefeas = ["time", "实际标记", "检测标记", "概率"]
    tpd = pd.read_csv(filepath)[acfeas]
    tpd.columns = prefeas
    tpd.to_csv(writefilepath, index=False)
