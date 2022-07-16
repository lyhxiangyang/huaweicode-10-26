import json
import os
import sys
import time
from typing import List

import pandas as pd

from hpc.l3l2utils.DefineData import FAULT_FLAG
from hpc.l3l2utils.ModelTrainLib import getNormalDectionJson, getTrainDectionJson, getAllDataFramesFromDectionJson, \
    getMemLeakPermin, getMaxflopsinio, getPgfreeThread, getddrc_ddwr_sumscope, changeModel, \
    getCPUTimeThread, getRandomcpuThreshold, getFreqDownThresholdpercent, getPowerThreshold, getTrainL3DectionJsonList, \
    getTrainL2DectionJsonList, mergeDataFrameDictList
from hpc.l3l2utils.ParsingJson import readJsonToDict, saveDictToJson

def isFlagsAnd(df: pd.DataFrame, labels: List[int]) -> bool:
    allablels = df[FAULT_FLAG].tolist()
    flags = [ilabels in allablels for ilabels in labels]
    return False not in flags
def isFlagsOr(df: pd.DataFrame, labels: List[int]) -> bool:
    allablels = df[FAULT_FLAG].tolist()
    flags = [ilabels in allablels for ilabels in labels]
    return True in flags

if __name__ == "__main__":
    startTime = time.perf_counter()
    # 读取训练配置文件
    modelconfigfilepath = os.path.join(sys.path[0], "modeltrainconfig.json")
    configJsonDict = readJsonToDict(*(os.path.split(modelconfigfilepath)))
    # 得到正常数据的json文件, 如果正常路径是null，那么返回None
    normalInputDict = getNormalDectionJson(configJsonDict)
    # 得到L3异常和L2异常数据的json文件列表 这是一个列表 如果路径是空，则返回[]
    L3abnormalInputDictList = getTrainL3DectionJsonList(configJsonDict)
    L2abnormalInputDictList = getTrainL2DectionJsonList(configJsonDict)

    # 得到L3异常和L2异常数据的DataFrame结构
    L3AllDataFrameDictList = [getAllDataFramesFromDectionJson(ijson) for ijson in L3abnormalInputDictList]
    L2AllDataFrameDictList = [getAllDataFramesFromDectionJson(ijson) for ijson in L2abnormalInputDictList]

    # 得到L3和L2对应的正常数据，默认情况下取第一个数据的faultFlag=0的数据作为正常数据, 如果列表长度为0 正常就是None
    L3NormalDataFrameDict = dict([(i, v[v[FAULT_FLAG] == 0]) for i, v in L3AllDataFrameDictList[0].items() if len(v) != 0]) if len(L3AllDataFrameDictList) != 0 else None
    L2NormalDataFrameDict = dict([(i, v[v[FAULT_FLAG] == 0]) for i, v in L2AllDataFrameDictList[0].items() if len(v) != 0]) if len(L2AllDataFrameDictList) != 0 else None
    if normalInputDict is not None: # 有正常路径的情况下,进行替换
        L3NormalDataFrameDict = getAllDataFramesFromDectionJson(normalInputDict)
        L2NormalDataFrameDict = getAllDataFramesFromDectionJson(normalInputDict)

    # 我们现在有L2的正常数据，L2的异常数据列表 需要将异常合并之后然后判断
    # 我们现在有L3的正常数据，L3的异常数据列表 需要将异常合并之后然后判断
    # 现在需要将L3AllDataFrameDictList和L2AllDataFrameDictList里面的数据进行合并，假如存在的话
    L3AllDataFrameDict = mergeDataFrameDictList(L3AllDataFrameDictList)
    L2AllDataFrameDict = mergeDataFrameDictList(L2AllDataFrameDictList)

    outputJsonDict = {}
    dataMeanDict = {}
    dataMeanDict["normalDataMean"] = {}
    dataMeanDict["normalDataMean"]["server"] = {}
    dataMeanDict["normalDataMean"]["topdown"] = {}
    dataMeanDict["normalDataMean"]["compute"] = {}

    if normalInputDict is not None:
        normalDataDictL3 = getAllDataFramesFromDectionJson(normalInputDict)
        normalDataDictL2 = getAllDataFramesFromDectionJson(normalInputDict)
    else:
        # 如果没有指定，那么normalInputDict来自于异常类型中的数值
        normalDataDictL3 = dict([(i, v[v[FAULT_FLAG] == 0]) for i, v in L3AllDataFrameDictList[0].items() if FAULT_FLAG in v.columns])
        normalDataDictL2 = normalDataDictL3.copy()
        # if len(L2AllDataFrameDictList) > 0:
        #     normalDataDictL2 = dict([(i, v[v[FAULT_FLAG] == 0]) for i, v in L3AllDataFrameDictList[0].items()])


    # 保证60内存泄露异常是存在的
    if isFlagsOr(L3AllDataFrameDict["server"], configJsonDict["memleaklabels"]):
        # 内存大小幅度的变化 - model -- ok
        memleakpermin = getMemLeakPermin(normalDataDictL3, L3AllDataFrameDict, configJsonDict)
        outputJsonDict["memleakpermin"] = memleakpermin

    # 保证50和90某个出现一个就行
    if isFlagsOr(L3AllDataFrameDict["server"], configJsonDict["memorybandwidthlabels"] + configJsonDict["cachegrablabels"]):
        # 得到mlops的变化幅度 主要是根据50 和 90的最大变化 - ok
        maxflopsinio = getMaxflopsinio(normalDataDictL3, L3AllDataFrameDict, configJsonDict)
        outputJsonDict["maxflopsinio"] = 0
        # 得到pgfree的变化幅度 - model - ok
        if isFlagsOr(L3AllDataFrameDict["server"], configJsonDict["memorybandwidthlabels"]):
            pgfree_thread = getPgfreeThread(normalDataDictL3, L3AllDataFrameDict, maxflopsinio, configJsonDict, dataMeanDict)
            outputJsonDict["pgfree_thread"] = pgfree_thread
        if isFlagsOr(L3AllDataFrameDict["server"], configJsonDict["cachegrablabels"]):
            # 得到读写指标的变化幅度 - model - ok
            ddrc_ddwr_sum_max = getddrc_ddwr_sumscope(normalDataDictL3, L3AllDataFrameDict, maxflopsinio, configJsonDict, dataMeanDict)
            outputJsonDict["ddrc_ddwr_sum_max"] = ddrc_ddwr_sum_max
    # 保证全CPU是存在的
    if isFlagsOr(L3AllDataFrameDict["server"], configJsonDict["allcpulabels"]):
        # 得到judgeCPUthread 主要是依据全CPU中判断的依据 - ok
        cpuTimeThread = getCPUTimeThread(normalDataDictL3, L3AllDataFrameDict, configJsonDict, dataMeanDict)
        outputJsonDict["abnormalCpuTimeThread"] = cpuTimeThread
    if isFlagsOr(L3AllDataFrameDict["server"], configJsonDict["randomcpulabels"] + configJsonDict["memorybandwidthlabels"] + configJsonDict["cachegrablabels"]):
        # 得到randomcpu
        randomcpuThreshold = getRandomcpuThreshold(normalDataDictL3, L3AllDataFrameDict, configJsonDict, dataMeanDict)
        outputJsonDict["randomcpuThreshold"] = randomcpuThreshold
    if L2AllDataFrameDict is not None:
        if isFlagsOr(L2AllDataFrameDict["server"], configJsonDict["l2labels"]):
            # 得到freqDownThresholdpercent
            freqDownThresholdpercent = getFreqDownThresholdpercent(normalDataDictL2, L2AllDataFrameDict, configJsonDict, dataMeanDict)
            outputJsonDict["freqDownThresholdpercent"] = freqDownThresholdpercent
        # 得到 power_threshold 通过111导致的power变化来改变
        outputJsonDict["power_threshold"] = 90
        if isFlagsOr(L2AllDataFrameDict["server"], [161]):
            power_threshold = getPowerThreshold(normalDataDictL2, L2AllDataFrameDict, configJsonDict, dataMeanDict)
            outputJsonDict["power_threshold"] = power_threshold

    if configJsonDict["testConfigPath"] is not None:
        testConfigDict = readJsonToDict(*(os.path.split(configJsonDict["testConfigPath"])))
        thresholds = ["memleakpermin", "maxflopsinio", "pgfree_thread", "ddrc_ddwr_sum_max",
                      "abnormalCpuTimeThread", "randomcpuThreshold", "freqDownThresholdpercent", "power_threshold"]
        for threshold in thresholds:
            if threshold in outputJsonDict.keys():
                testConfigDict[threshold] = outputJsonDict[thresholds]
        data_means = ["server", "compute", "process", "nic", "ping", "topdown"]
        for mean in data_means:
            if mean in dataMeanDict["normalDataMean"].keys():
                testConfigDict["normalDataMean"][mean] = dataMeanDict["normalDataMean"][mean]
        with open(configJsonDict["testConfigPath"], 'w', encoding='utf8') as f:
            json.dump(testConfigDict, f, indent=4, ensure_ascii=False)

    if configJsonDict["debugpath"] is not None:
        tpath = os.path.join(configJsonDict["debugpath"], "result")
        saveDictToJson(outputJsonDict, tpath, "parameter.json")
        saveDictToJson(dataMeanDict, tpath, "datamean.json")
    if configJsonDict["outputpath"] is not None:
        tpath = os.path.join(configJsonDict["outputpath"])
        saveDictToJson(outputJsonDict, tpath, "parameter.json")
        saveDictToJson(dataMeanDict, tpath, "datamean.json")


    endTime1 = time.perf_counter()
    print('Running time: %s Seconds' % (endTime1 - startTime))
