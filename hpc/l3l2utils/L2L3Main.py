import os
from typing import List, Dict
import pandas as pd

from hpc.Classifiers import select_and_pred
from hpc.l3l2utils.DataFrameOperation import mergeDataFrames, mergeinnerTwoDataFrame, mergeouterPredictResult
from hpc.l3l2utils.DataFrameSaveRead import saveDFListToFiles, savepdfile
from hpc.l3l2utils.DataOperation import changeTimeToFromPdlists
from hpc.l3l2utils.DefineData import TIME_COLUMN_NAME, FAULT_FLAG, MODEL_TYPE
from hpc.l3l2utils.FeatureExtraction import differenceProcess, differenceServer, standardLists, extractionProcessPdLists, \
    extractionServerPdLists
from hpc.l3l2utils.ParsingJson import readJsonToDict, getServerPdFromJsonDict, getProcessPdFromJsonDict, \
    getL2PdFromJsonDict, getNetworkPdFromJsonDict, getNormalServerMean, getNormalProcessMean, getNormalL2Mean, \
    getNormalNetworkMean, saveDictToJson, getPingPdFromJsonDict
from hpc.l3l2utils.l3l2detection import fixFaultFlag, fixIsolatedPointPreFlag, getDetectionProbability, getTimePeriodInfo, \
    analysePredictResult
from hpc.l3l2utils.modelpred import detectL3CPUAbnormal, detectL3MemLeakAbnormal, detectL3BandWidthAbnormal, predictTemp, \
    detectNetwork_TXHangAbnormal

"""
time faultFlag preFlag
"""


def makeL2networkresultMergedByMin(l2networkpd: pd.DataFrame) -> pd.DataFrame:
    def getrightflag(x):
        x = list(x)
        maxlabel = max(x, key=x.count)
        return maxlabel

    delpd = l2networkpd.groupby(TIME_COLUMN_NAME, as_index=False).agg([getrightflag])
    delpd: pd.DataFrame
    # delpd.reset_index(inplace=True)
    delpd.columns = [i[0] for i in delpd.columns]
    delpd.reset_index(inplace=True)
    return delpd


def ThransferRightLabels(x: List[int]):
    C = [0, 111, 121, 131, 132, 141]
    y = [C[i] if 0 <= i <= 5 else i for i in x]
    return y


def add_cpu_column(pds: List[pd.DataFrame]):
    for ipd in pds:
        ipd['cpu'] = ipd['user'] + ipd['system']


"""
将数据进行特征提取
"""


def FeatureextractionData(inputDict: Dict):
    print("将数据从文件中读取".center(40, "*"))
    detectionJson = readJsonToDict(*(os.path.split(inputDict["predictdirjsonpath"])))
    predictserverpds = getServerPdFromJsonDict(sdict=detectionJson)
    predictprocesspds = getProcessPdFromJsonDict(sdict=detectionJson)
    predictl2pds = getL2PdFromJsonDict(sdict=detectionJson)
    predictnetworkpds = getNetworkPdFromJsonDict(sdict=detectionJson)
    predictpingpds = getPingPdFromJsonDict(sdict=detectionJson)

    print("对读取到的原始数据进行差分".format(40, "*"))
    # 对异常数据进行差分处理之后，得到cpu特征值
    predictprocesspds = differenceProcess(predictprocesspds, inputDict["process_accumulate_feature"])
    add_cpu_column(predictprocesspds)
    # 对异常server服务数据进行差分处理之后，得到一些指标
    predictserverpds = differenceServer(predictserverpds, inputDict["server_accumulate_feature"])
    predictnetworkpds = differenceServer(predictnetworkpds, inputDict["network_accumulate_feature"])
    predictl2pds = differenceServer(predictl2pds, inputDict["l2_accumulate_feature"])
    predictpingpds = differenceServer(predictpingpds, inputDict["ping_accumulate_feature"])
    inputDict["process_feature"] = ["cpu"]

    print("对正常数据的各个指标求平均值".center(40, "*"))
    normalserver_meanvalue = getNormalServerMean(detectionJson, predictserverpds, predictprocesspds,
                                                 inputDict["server_feature"],
                                                 datanumber=inputDict["meanNormalDataNumber"])
    normalprocess_meanvalue = getNormalProcessMean(detectionJson, predictprocesspds, inputDict["process_feature"],
                                                   datanumber=inputDict["meanNormalDataNumber"])
    normall2_meanvalue = getNormalL2Mean(detectionJson, predictl2pds, inputDict["l2_feature"],
                                         datanumber=inputDict["meanNormalDataNumber"])
    normalnetwork_meanvalue = getNormalNetworkMean(detectionJson, predictnetworkpds, inputDict["network_feature"],
                                                   datanumber=inputDict["meanNormalDataNumber"])
    # ---- 不对ping求平均值
    # 将数据进行保存
    if inputDict["spath"] is not None:
        tpath = os.path.join(inputDict["spath"], "1.正常数据的平均值")
        savepdfile(normalserver_meanvalue, tpath, "meanvalue_server.csv")
        savepdfile(normalprocess_meanvalue, tpath, "meanvalue_process.csv")
        savepdfile(normall2_meanvalue, tpath, "meanvalue_l2.csv")
        savepdfile(normalnetwork_meanvalue, tpath, "meanvalue_network.csv")
    # ========================================

    print("标准化要预测的process和server数据".center(40, "*"))
    standard_server_pds = standardLists(pds=predictserverpds, standardFeatures=inputDict["server_feature"],
                                        meanValue=normalserver_meanvalue, standardValue=100)
    standard_process_pds = standardLists(pds=predictprocesspds, standardFeatures=inputDict["process_feature"],
                                         meanValue=normalprocess_meanvalue, standardValue=60)
    standard_l2_pds = standardLists(pds=predictl2pds, standardFeatures=inputDict["l2_feature"],
                                    meanValue=normall2_meanvalue, standardValue=100)
    standard_network_pds = standardLists(pds=predictnetworkpds, standardFeatures=inputDict["network_feature"],
                                         meanValue=normalnetwork_meanvalue, standardValue=100)
    # -----不对ping进行标准化
    standard_ping_pds = predictpingpds
    # 将数据进行保存
    if inputDict["spath"] is not None:
        tpath = os.path.join(inputDict["spath"], "2.标准化数据存储")
        saveDFListToFiles(os.path.join(tpath, "server_standard"), standard_server_pds)
        saveDFListToFiles(os.path.join(tpath, "process_standard"), standard_process_pds)
        saveDFListToFiles(os.path.join(tpath, "l2_standard"), standard_l2_pds)
        saveDFListToFiles(os.path.join(tpath, "network_standard"), standard_network_pds)
        saveDFListToFiles(os.path.join(tpath, "ping_standard"), standard_ping_pds)

    print("将数据的时间进行统一化处理".center(40, "*"))
    standard_server_pds = changeTimeToFromPdlists(standard_server_pds, isremoveDuplicate=True)
    standard_process_pds = changeTimeToFromPdlists(standard_process_pds)
    standard_l2_pds = changeTimeToFromPdlists(standard_l2_pds, isremoveDuplicate=True)
    standard_network_pds = changeTimeToFromPdlists(standard_network_pds, isremoveDuplicate=True)
    standard_ping_pds = changeTimeToFromPdlists(standard_ping_pds, isremoveDuplicate=False)

    print("process、server、l2、network特征处理".center(40, "*"))
    extraction_process_pds = extractionProcessPdLists(standard_process_pds,
                                                      extractFeatures=inputDict["process_feature"],
                                                      windowsSize=3)
    extraction_server_pds = extractionServerPdLists(standard_server_pds, extractFeatures=inputDict["server_feature"],
                                                    windowsSize=3)
    extraction_l2_pds = extractionServerPdLists(standard_l2_pds, extractFeatures=inputDict["l2_feature"], windowsSize=3)
    extraction_network_pds = extractionServerPdLists(standard_network_pds, extractFeatures=inputDict["network_feature"],
                                                     windowsSize=3)
    # ----- 不对ping数据进行特征处理
    extraction_ping_pds = standard_ping_pds
    # 将数据进行保存
    if inputDict["spath"] is not None:
        tpath = os.path.join(inputDict["spath"], "3.特征提取")
        saveDFListToFiles(os.path.join(tpath, "server_extraction"), extraction_server_pds)
        saveDFListToFiles(os.path.join(tpath, "process_extraction"), extraction_process_pds)
        saveDFListToFiles(os.path.join(tpath, "l2_extraction"), extraction_l2_pds)
        saveDFListToFiles(os.path.join(tpath, "network_extraction"), extraction_network_pds)
    return extraction_server_pds, extraction_process_pds, extraction_l2_pds, extraction_network_pds, extraction_ping_pds


"""
返回一个L2L3层合并之后的数据
"""


def detectionL2L3Data(inputDict: Dict, allserverpds: pd.DataFrame, allprocesspds: pd.DataFrame,
                      alll2pds: pd.DataFrame, allnetworkpds: pd.DataFrame, allpingpds: pd.DataFrame) -> pd.DataFrame:
    # 需要用到的特征值
    resfeatures = [TIME_COLUMN_NAME, FAULT_FLAG, "preFlag"]

    print("对L3层CPU异常进行预测".center(40, "*"))
    tpath = None
    if inputDict["spath"] is not None:
        tpath = os.path.join(inputDict["spath"], "4.CPU异常检测中间文件")
    l3cpuresult = pd.DataFrame()
    l3cpuresult[TIME_COLUMN_NAME] = allserverpds[TIME_COLUMN_NAME]
    l3cpuresult[FAULT_FLAG] = allserverpds[FAULT_FLAG]
    l3cpuresult["preFlag"] = detectL3CPUAbnormal(allserverpds=allserverpds, allprocesspds=allprocesspds, spath=tpath,
                                                 modelfilepath=inputDict["processcpu_modelpath"],
                                                 modeltype=inputDict["processcpu_modeltype"],
                                                 processFeatures=inputDict["process_feature"])

    print("对L3层内存泄露进行检测".center(40, "*"))
    l3memleakresult = pd.DataFrame()
    l3memleakresult[TIME_COLUMN_NAME] = allserverpds[TIME_COLUMN_NAME]
    l3memleakresult[FAULT_FLAG] = allserverpds[FAULT_FLAG]
    l3memleakresult["preFlag"] = detectL3MemLeakAbnormal(allserverpds=allserverpds,
                                                         modelfilepath=inputDict["servermemory_modelpath"],
                                                         modeltype=inputDict["servermemory_modeltype"],
                                                         serverFeatures=inputDict["server_feature"])

    print("对L3层内存带宽进行检测".center(40, "*"))
    l3BandWidthResult = pd.DataFrame()
    l3BandWidthResult[TIME_COLUMN_NAME] = allserverpds[TIME_COLUMN_NAME]
    l3BandWidthResult[FAULT_FLAG] = allserverpds[FAULT_FLAG]
    l3BandWidthResult["preFlag"] = detectL3BandWidthAbnormal(allserverpds=allserverpds,
                                                             modelfilepath=inputDict["serverbandwidth_modelpath"],
                                                             modeltype=inputDict["serverbandwidth_modeltype"],
                                                             serverFeatures=inputDict["server_feature"])

    print("对L2层数据进行预测".center(40, "*"))
    l2_serverpds = mergeinnerTwoDataFrame(lpd=alll2pds, rpd=allserverpds)  # 根据时间得到l2的合并结果

    print("对L2机器封顶进行预测".center(40, "#"))
    l2machinepowerresult = pd.DataFrame()
    l2machinepowerresult[TIME_COLUMN_NAME] = l2_serverpds[TIME_COLUMN_NAME]
    l2machinepowerresult[FAULT_FLAG] = l2_serverpds[FAULT_FLAG]
    l2machinepowerresult["preFlag"] = ThransferRightLabels(
        select_and_pred(l2_serverpds, MODEL_TYPE[inputDict["power_machine_modeltype"]],
                        saved_model_path=inputDict["power_machine_modelpath"]))

    print("对L2机柜封顶进行预测".center(40, "#"))
    l2cabinetpowerresult = pd.DataFrame()
    l2cabinetpowerresult[TIME_COLUMN_NAME] = l2_serverpds[TIME_COLUMN_NAME]
    l2cabinetpowerresult[FAULT_FLAG] = l2_serverpds[FAULT_FLAG]
    l2cabinetpowerresult["preFlag"] = ThransferRightLabels(
        select_and_pred(l2_serverpds, MODEL_TYPE[inputDict["power_cabinet_modeltype"]],
                        saved_model_path=inputDict["power_cabinet_modelpath"]))

    print("对温度进行预测".center(40, "#"))
    l2temperamentresult = pd.DataFrame()
    l2temperamentresult[TIME_COLUMN_NAME] = l2_serverpds[TIME_COLUMN_NAME]
    l2temperamentresult[FAULT_FLAG] = l2_serverpds[FAULT_FLAG]
    # l2temperamentresult["preFlag"] = ThransferRightLabels(select_and_pred(l2_serverpds, MODEL_TYPE[tempertature_modeltype], saved_model_path=temperature_modelpath))
    l2temperamentresult["preFlag"] = ThransferRightLabels(predictTemp(model_path=inputDict["temperature_modelpath"],
                                                                      model_type=MODEL_TYPE[
                                                                          inputDict["tempertature_modeltype"]],
                                                                      data=l2_serverpds))

    print("对网络异常1进行预测 TX_Hang".center(40, "#"))
    # REPORT_TIME = "time"
    # l2networkresult1 = pd.DataFrame()
    # l2networkresult1[TIME_COLUMN_NAME] = allnetworkpds[REPORT_TIME]
    # l2networkresult1[FAULT_FLAG] = allnetworkpds[FAULT_FLAG]
    # l2networkresult1 = makeL2networkresultMergedByMin(l2networkresult1)
    l2networkresult1 = detectNetwork_TXHangAbnormal(allpingpds)

    print("对网络异常2 pfc进行预测".center(40, "#"))
    l2networkresult2 = pd.DataFrame()
    l2networkresult2[TIME_COLUMN_NAME] = allnetworkpds[TIME_COLUMN_NAME]
    l2networkresult2[FAULT_FLAG] = allnetworkpds[FAULT_FLAG]
    l2networkresult2["preFlag"] = ThransferRightLabels(
        select_and_pred(allnetworkpds, MODEL_TYPE[inputDict["network_model1type"]],
                        saved_model_path=inputDict["network_model1path"]))
    l2networkresult2 = makeL2networkresultMergedByMin(l2networkresult2)

    print("将L2 L3 Network数据合并分析".center(40, "*"))
    allresultspd = mergeouterPredictResult(
        [l3cpuresult, l3memleakresult, l3BandWidthResult, l2machinepowerresult, l2cabinetpowerresult,
         l2temperamentresult, l2networkresult1, l2networkresult2])

    print("对结果进行优化".center(40, "*"))
    allresultspd = fixFaultFlag(allresultspd)
    allresultspd = fixIsolatedPointPreFlag(allresultspd)

    print("增加此时间点是否预测正确".center(40, "*"))
    isrightLists = [1 if
                    allresultspd[FAULT_FLAG][i] != 0 and
                    (allresultspd[FAULT_FLAG][i] in allresultspd["preFlag"][i] or
                     allresultspd[FAULT_FLAG][i] // 10 * 10 in allresultspd["preFlag"][i]) else 0 for i in
                    range(0, len(allresultspd))]
    allresultspd["isright"] = isrightLists

    print("增加概率".center(40, "*"))
    allresultspd["probability"] = getDetectionProbability(allresultspd["preFlag"].tolist())
    return allresultspd


"""
通过preFlag得到概率，其中preFlag每个元素是list
"""

"""
得到一个json形式的Dict
"""


def outputJsonFromDetection(l2l3predetectresultpd: pd.DataFrame) -> Dict:
    l2l3predetectresultpd = l2l3predetectresultpd.copy()
    outputDict = {"error": getTimePeriodInfo(l2l3predetectresultpd, "preFlag", "probability")}
    return outputDict



def saveoutputJsonFilename(inputDict: Dict, outputJsonDict):
    # =================================  默认情况下，在当前目录生成output.json
    outputJsonFilename = "output.json"
    if "resultsavepath" in inputDict and inputDict["resultsavepath"] is not None: # 路径不能为空，如果为空，默认为当前目录
        resultsavepath = os.path.abspath(inputDict["resultsavepath"])
        # if inputDict["resultsavepath"] == ".":
        #     resultsavepath = sys.path[0]
        # print("绝对路径：{}".format(resultsavepath))
        if not os.path.exists(resultsavepath):
            print("输出结果路径:{}不存在".format(resultsavepath))
            exit(1)
    else:
        return # 不进行保存
    if "outputJsonFilename" in inputDict and inputDict["outputJsonFilename"] is not None:
        outputJsonFilename = inputDict["outputJsonFilename"]
    saveDictToJson(outputJsonDict, resultsavepath, outputJsonFilename)

"""
从输入json文件中读取信息，进行检测
"""


def detectionFromInputDict(inputDict: Dict) -> Dict:
    # =====================将数据进行特征提取
    extraction_server_pds, extraction_process_pds, extraction_l2_pds, extraction_network_pds, extraction_ping_pds = FeatureextractionData(
        inputDict)
    # =====================数据合并
    allserverpds = mergeDataFrames(extraction_server_pds)
    allprocesspds = mergeDataFrames(extraction_process_pds)
    alll2pds = mergeDataFrames(extraction_l2_pds)
    allnetworkpds = mergeDataFrames(extraction_network_pds)
    allpingpds = mergeDataFrames(extraction_ping_pds)

    print("对L3 L2层的数据进行预测".center(40, "*"))
    l2l3predetectresultpd = detectionL2L3Data(inputDict, allserverpds, allprocesspds, alll2pds, allnetworkpds,
                                              allpingpds)
    tpath = None
    if inputDict["spath"] is not None:
        tpath = os.path.join(inputDict["spath"], "5.准确率结果分析")
    # 对预测结果进行分析
    print("对预测结果进行准确率及其他分析".center(40, "*"))
    analysePredictResult(l2l3predetectresultpd, spath=tpath, windowsize=3)
    print("对预测结果进行时间段分析，输出时间文件".center(40, "*"))
    outputDict = outputJsonFromDetection(l2l3predetectresultpd)

    # =============================将信息保存到磁盘
    if inputDict["spath"] is not None:
        tpath = os.path.join(inputDict["spath"], "6.L2L3总的预测结果")
        savepdfile(l2l3predetectresultpd, tpath, "总预测结果.csv")
        saveDictToJson(outputDict, tpath, "output.json")


    # ============================生成outputjson
    saveoutputJsonFilename(inputDict, outputDict)






    return outputDict







































































































































































































