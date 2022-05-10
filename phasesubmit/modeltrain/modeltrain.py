import os
import sys
import time
from typing import Dict, List

import pandas as pd

from hpc.l3l2utils.DataFrameOperation import mergeDataFrames, mergeProceeDF, smoothseries, getSeriesFrequencyMeanLists
from hpc.l3l2utils.DataFrameSaveRead import savepdfile
from hpc.l3l2utils.DataOperation import changeTimePdsToStrFromInt, changeTimeToFromPdlists
from hpc.l3l2utils.DefineData import usefulFeatures, TIME_COLUMN_NAME, FAULT_FLAG
from hpc.l3l2utils.FeatureExtraction import differenceProcess, differenceServer
from hpc.l3l2utils.L2L3Main import detectionFromInputDict
from hpc.l3l2utils.ParsingJson import readJsonToDict, getServerPdFromJsonDict, getProcessPdFromJsonDict, \
    getL2PdFromJsonDict, getNetworkPdFromJsonDict, getPingPdFromJsonDict, getTopdownPdFromJsonDict
from hpc.l3l2utils.PreProcessConfig import preproccessConfigfile

"""
2. memleakmin  内存泄露的最小值
3. pgfree_thread pgfree的变化幅度
4. ddrc_ddwr_sum_max的变化幅度
5. maxflopsinio mflops的变化幅度
"""

"""
函数功能：将modeltrain的config中的正常训练数据和异常训练数据取得
"""
def getNormalDectionJson(modeltrainconfig: Dict, normalDectionjson: Dict = None) -> Dict:
    if normalDectionjson is None and modeltrainconfig["trainNormalDataPath"] is not None:
        normalDectionjson=readJsonToDict(*(os.path.split(modeltrainconfig["trainNormalDataPath"])))
    return normalDectionjson
"""
函数功能：将modeltrain的config中的正常训练数据和异常训练数据取得
"""
def getTrainDectionJson(modeltrainconfig: Dict, abnormalDectionjson: Dict = None) -> Dict:
    if abnormalDectionjson is None:
        abnormalDectionjson=readJsonToDict(*(os.path.split(modeltrainconfig["trainAbnormalDataPath"])))
    return abnormalDectionjson
"""
函数功能：从DectionJson中获得需要的数据并且进行简单的时间和差分处理
"""
def getAllDataFramesFromDectionJson(dataDectionJson: Dict, isTimeisStr: bool=True)->Dict:
    if dataDectionJson is None:
        return None
    predictserverpds = getServerPdFromJsonDict(sdict=dataDectionJson)
    predictprocesspds = getProcessPdFromJsonDict(sdict=dataDectionJson)
    predictl2pds = getL2PdFromJsonDict(sdict=dataDectionJson)
    predictnetworkpds = getNetworkPdFromJsonDict(sdict=dataDectionJson)
    predictpingpds = getPingPdFromJsonDict(sdict=dataDectionJson)
    predicttopdwnpds = getTopdownPdFromJsonDict(sdict=dataDectionJson)
    if not isTimeisStr:
        changeTimePdsToStrFromInt(predictserverpds)
        changeTimePdsToStrFromInt(predictprocesspds)
        changeTimePdsToStrFromInt(predictl2pds)
        changeTimePdsToStrFromInt(predictnetworkpds)
        changeTimePdsToStrFromInt(predictpingpds)
        changeTimePdsToStrFromInt(predicttopdwnpds)
    # 时间统一处理
    predictserverpds = changeTimeToFromPdlists(predictserverpds, isremoveDuplicate=True)
    predictprocesspds = changeTimeToFromPdlists(predictprocesspds)
    predictl2pds = changeTimeToFromPdlists(predictl2pds, isremoveDuplicate=True)
    predictnetworkpds = changeTimeToFromPdlists(predictnetworkpds, isremoveDuplicate=True)
    predictpingpds = changeTimeToFromPdlists(predictpingpds, isremoveDuplicate=False)
    predicttopdwnpds = changeTimeToFromPdlists(predicttopdwnpds, isremoveDuplicate=True)

    # 差分处理
    # 对异常数据进行差分处理之后，得到cpu特征值 process数据会去掉前三min和后三min的数值
    predictprocesspds = differenceProcess(predictprocesspds, usefulFeatures["process_diff"])
    # 对异常server服务数据进行差分处理之后，得到一些指标
    predictserverpds = differenceServer(predictserverpds, usefulFeatures["server_diff"])
    predictnetworkpds = differenceServer(predictnetworkpds, usefulFeatures["network_diff"])
    predictl2pds = differenceServer(predictl2pds, usefulFeatures["compute_diff"])
    predictpingpds = differenceServer(predictpingpds, usefulFeatures["ping_diff"])
    predicttopdwnpds = differenceServer(predicttopdwnpds, usefulFeatures["topdown_diff"])
    resDict = {
        "server": mergeDataFrames(predictserverpds),
        "process": mergeDataFrames(predictprocesspds),
        "network": mergeDataFrames(predictnetworkpds),
        "compute": mergeDataFrames(predictl2pds),
        "ping": mergeDataFrames(predictpingpds),
        "topdown": mergeDataFrames(predicttopdwnpds)
    }
    return resDict

"""
函数功能：从表格中提取标签为abnormalType的异常值
"""
def abstractAbnormalData(datadf: pd.DataFrame, abnormalType: List[int], labelFlag: str=FAULT_FLAG):
    return datadf[datadf[labelFlag].isin(abnormalType)]

"""
得到
"""
def getMemLeakPermin(normalfilepdDict: Dict, abnormalfilepdDict: Dict, modelconfigJson: Dict=None, )->int:
    # 对内存的差值处理必须根据进程pid号进行处理
    # 保证内存的指标和pid号的指标长度是一样的
    def diffmemoryseries(memseries: pd.Series, pidseries: pd.Series):
        assert len(memseries) == len(pidseries)
        df = pd.DataFrame(data={
            "mem": memseries,
            "pid": pidseries
        })
        # 直接写个for循环
        reslists = []
        winsize = 5
        for ipid, idf in df.groupby("pid", sort=False):
            other_mem_smooth = smoothseries(idf["mem"], windows=winsize)
            other_mem_smooth_diff = other_mem_smooth.diff(1).fillna(0)
            reslists.extend(other_mem_smooth_diff.tolist())
        return pd.Series(data=reslists)


    # 根据server和process中的memory数据得到内存的变化量
    def getMemory(serverpd: pd.DataFrame, processpd: pd.DataFrame)->pd.DataFrame:
        mergeprocesspd = mergeProceeDF(processpd, sumFeatures=["rss"])
        # 将两者合并
        pspd = pd.merge(left=serverpd, right=mergeprocesspd, left_on=TIME_COLUMN_NAME, right_on=TIME_COLUMN_NAME, suffixes=("", "_y"))

        pspd.fillna(0, inplace=True) # 认为进程不在的时候其数据为0

        servermem = pspd["mem_used"]
        processmem = pspd["rss"]
        othermem = servermem - processmem

        othermemdiff=diffmemoryseries(othermem, pspd["pid"]) / 1000000
        # 返回将带有时间与内存
        respd = pd.DataFrame()
        respd[TIME_COLUMN_NAME] = pspd[TIME_COLUMN_NAME]
        respd["server_mem_used"] = servermem
        respd["process_mem_used"] = processmem
        respd["mem_used_mean"] = othermemdiff
        respd[FAULT_FLAG] = serverpd[FAULT_FLAG]
        return respd
    # memleak从abnormalleak得到
    processpd = abnormalfilepdDict["process"].copy()
    serverpd = abnormalfilepdDict["server"].copy()
    memorypd = getMemory(serverpd=serverpd, processpd=processpd)
    memorypd60 = abstractAbnormalData(memorypd, [60,63,64,65])
    memory_mean = getSeriesFrequencyMeanLists(memorypd60, ["mem_used_mean"])
    memorypd["memory_mean"] = memory_mean
    memorypd["memory_mean_p80"] = memory_mean * 0.8
    if modelconfigJson["debugpath"] is not None:
        tpath = os.path.join(modelconfigJson["debugpath"], "memleak60")
        savepdfile(memorypd, tpath, "memleakdebug.csv")
    return memory_mean



if __name__ == "__main__":
    startTime = time.perf_counter()
    modelconfigfilepath = os.path.join(sys.path[0], "modeltrainconfig.json")
    configJsonDict = readJsonToDict(*(os.path.split(modelconfigfilepath)))
    normalInputDict = getNormalDectionJson(configJsonDict)
    abnormalInputDict = getTrainDectionJson(configJsonDict)

    normalDataDict = getAllDataFramesFromDectionJson(normalInputDict)
    abnormalDataDict = getAllDataFramesFromDectionJson(abnormalInputDict)

    getMemLeakPermin(normalDataDict, abnormalDataDict, configJsonDict)
    endTime1 = time.perf_counter()
    print('Running time: %s Seconds' % (endTime1 - startTime))
