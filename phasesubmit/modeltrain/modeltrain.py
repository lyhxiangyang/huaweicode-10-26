import os
import sys
import time
from typing import Dict

from hpc.classifiers.ModelChange import change_threshold
from hpc.l3l2utils.DefineData import MODEL_TYPE
from hpc.l3l2utils.ModelTrainLib import getNormalDectionJson, getTrainDectionJson, getAllDataFramesFromDectionJson, \
    getMemLeakPermin, getMaxflopsinio, getPgfreeThread, getddrc_ddwr_sumscope, changeModel
from hpc.l3l2utils.ParsingJson import readJsonToDict, saveDictToJson

if __name__ == "__main__":
    startTime = time.perf_counter()
    modelconfigfilepath = os.path.join(sys.path[0], "modeltrainconfig.json")
    configJsonDict = readJsonToDict(*(os.path.split(modelconfigfilepath)))
    normalInputDict = getNormalDectionJson(configJsonDict)
    abnormalInputDict = getTrainDectionJson(configJsonDict)

    normalDataDict = getAllDataFramesFromDectionJson(normalInputDict)
    abnormalDataDict = getAllDataFramesFromDectionJson(abnormalInputDict)
    # 内存大小幅度的变化 - model
    memleakpermin = getMemLeakPermin(normalDataDict, abnormalDataDict, configJsonDict)
    # 得到mlops的变化幅度
    maxflopsinio = getMaxflopsinio(normalDataDict, abnormalDataDict, configJsonDict)
    # 得到pgfree的变化幅度 - model
    pgfree_thread = getPgfreeThread(normalDataDict, abnormalDataDict, maxflopsinio, configJsonDict)
    # 得到读写指标的变化幅度 - model
    ddrc_ddwr_sum_max = getddrc_ddwr_sumscope(normalDataDict, abnormalDataDict, maxflopsinio, configJsonDict)

    outputJsonDict = {
        "memleakpermin": memleakpermin,
        "maxflopsinio": maxflopsinio,
        "pgfree_thread": pgfree_thread,
        "ddrc_ddwr_sum_max": ddrc_ddwr_sum_max,
    }
    if configJsonDict["isChangeModel"]:
        changeModel(configJsonDict, outputJsonDict)
    if configJsonDict["debugpath"] is not None:
        tpath = os.path.join(configJsonDict["debugpath"], "result")
        saveDictToJson(outputJsonDict, tpath, "parameter.json")
    if configJsonDict["outputpath"] is not None:
        tpath = os.path.join(configJsonDict["outputpath"])
        saveDictToJson(outputJsonDict, tpath, "parameter.json")


    endTime1 = time.perf_counter()
    print('Running time: %s Seconds' % (endTime1 - startTime))
