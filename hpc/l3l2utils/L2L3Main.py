import os
from typing import List, Dict, Tuple, Union, Any
import pandas as pd
from pandas import Series, DataFrame
from pandas.core.generic import NDFrame

from hpc.classifiers.ModelPred import select_and_pred
from hpc.l3l2utils.DataFrameOperation import mergeDataFrames, mergeinnerTwoDataFrame, mergeouterPredictResult
from hpc.l3l2utils.DataFrameSaveRead import saveDFListToFiles, savepdfile
from hpc.l3l2utils.DataOperation import changeTimeToFromPdlists, removeProcessUselessData, getRunHPCTimepdsFromProcess, \
    changeTimePdsToStrFromInt
from hpc.l3l2utils.DefineData import TIME_COLUMN_NAME, FAULT_FLAG, MODEL_TYPE
from hpc.l3l2utils.FeatureExtraction import differenceProcess, differenceServer, standardLists, \
    extractionProcessPdLists, \
    extractionServerPdLists
from hpc.l3l2utils.ParsingJson import readJsonToDict, getServerPdFromJsonDict, getProcessPdFromJsonDict, \
    getL2PdFromJsonDict, getNetworkPdFromJsonDict, getNormalServerMean, getNormalProcessMean, getNormalL2Mean, \
    getNormalNetworkMean, saveDictToJson, getPingPdFromJsonDict, getTopdownPdFromJsonDict, getNormalTopdownMean, \
    getMeanFromNumberDataFrom
from hpc.l3l2utils.l3l2detection import fixFaultFlag, fixIsolatedPointPreFlag, getDetectionProbability, \
    getTimePeriodInfo, \
    analysePredictResult
from hpc.l3l2utils.modelpred import detectL3CPUAbnormal, detectL3MemLeakAbnormal, detectL3BandWidthAbnormal, \
    predictTemp, \
    detectNetwork_TXHangAbnormal, predictL2_CPUDown, predictCacheGrab, predictCabinet_PowerCapping, \
    predictServer_PowerCapping

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
        ipd['cpu'] = ipd["usr_cpu"] + ipd["kernel_cpu"]


"""
对topdown数据列表进行处理
主要是对ddrc_rd 以及 ddrc_wr进行处理
spath 需要往里面写入topdwon的平均值， 平均值可以有多种获得方法，1. 前三个的平均值  2.前10个的平均值 3. 去除异常之后的平均值
"""

"""
处理对topdown数据
"""


def processTopdownList(detectJson: Dict, predicttopdwnpds: List[pd.DataFrame]) -> List[pd.DataFrame]:
    def proceeOneTopdownPd(itopdownpd: pd.DataFrame):
        # 这个会对mflops进行处理，然后
        if "mflops_change" not in itopdownpd.columns.array:
            cname = "mflops"
            itopdownpd = removeUselessDataFromTopdownList([itopdownpd])[0]
            itopdownpd[cname] = itopdownpd[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
            itopdownpd[cname] = itopdownpd[cname].rolling(window=5, center=True, min_periods=1).mean()
            mflops_mean = getNormalTopdownMean(detectJson, [itopdownpd], [cname], datanumber=10)[cname]
            mflops_change = itopdownpd[cname].apply(lambda x: (mflops_mean - x) / mflops_mean if x < mflops_mean else 0)
            itopdownpd["mflops_change"] = mflops_change
        mflops_change = itopdownpd["mflops_change"]


        # 对ddrc_rd进行滑动窗口处理
        rd_cname = "ddrc_rd"
        itopdownpd[rd_cname] = itopdownpd[rd_cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
        itopdownpd[rd_cname] = itopdownpd[rd_cname].rolling(window=5, center=True, min_periods=1).mean()
        ddrc_rd_mean = getNormalTopdownMean(detectJson, [itopdownpd], [rd_cname], datanumber=10)[rd_cname]
        itopdownpd[rd_cname] = itopdownpd[rd_cname] + ddrc_rd_mean * mflops_change

        # 对ddrc_rd进行滑动窗口处理
        wr_cname = "ddrc_wr"
        itopdownpd[wr_cname] = itopdownpd[wr_cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
        itopdownpd[wr_cname] = itopdownpd[wr_cname].rolling(window=5, center=True, min_periods=1).mean()
        ddrc_rd_mean = getNormalTopdownMean(detectJson, [itopdownpd], [wr_cname], datanumber=10)[wr_cname]
        itopdownpd[wr_cname] = itopdownpd[wr_cname] + ddrc_rd_mean * mflops_change

        # 对rd_wr_sum进行结合 减去平均值  阈值与6000比较
        rd_wr_cname = "ddrc_ddwr_sum"
        itopdownpd[rd_wr_cname] = itopdownpd[rd_cname] + itopdownpd[wr_cname]
        itopdownpd[rd_wr_cname] = itopdownpd[rd_wr_cname].rolling(window=5, center=True, min_periods=1).median()
        rd_wr_sum_mean = getNormalTopdownMean(detectJson, [itopdownpd], [rd_wr_cname], datanumber=10)[rd_wr_cname]
        itopdownpd[rd_wr_cname] = itopdownpd[rd_wr_cname] - rd_wr_sum_mean
        return itopdownpd

        # 对ddrc_wr进行滑动窗口处理

        # # 对mflops进行分析
        # itopdownpd["mflops_sliding"] = itopdownpd["mflops"].rolling(window=5, center=True, min_periods=1).agg(
        #     "max").astype("int")
        # mflops_mean = itopdownpd["mflops_sliding"][0:3].mean() # todo
        #
        # print("mflops平均值：{}".format(mflops_mean))
        # mflops_change = itopdownpd["mflops_sliding"].apply(
        #     lambda x: (mflops_mean - x) / mflops_mean if x <= mflops_mean else 0)  # 如果是-20% 那么对应的值应该增加20%
        #
        # # 对指标进行补偿性分析
        # cname = "ddrc_rd"
        # cname_sliding = cname + "_sliding"
        # itopdownpd[cname_sliding] = itopdownpd[cname].rolling(window=5, center=True, min_periods=1).agg("max").astype(
        #     "int")
        # ddrc_rd_mean = itopdownpd[cname_sliding][0:3].mean()  # 得到一个正常值 todo
        # print("{}平均值：{}".format(cname, ddrc_rd_mean))
        # itopdownpd[cname_sliding + "_recover"] = itopdownpd[cname_sliding] + ddrc_rd_mean * mflops_change
        # itopdownpd[cname_sliding + "_recover_sliding"] = itopdownpd[cname_sliding + "_recover"].rolling(window=5,
        #                                                                                                 center=True,
        #                                                                                                 min_periods=1).agg(
        #     "max").astype("int")
        #
        # cname = "ddrc_wr"
        # cname_sliding = cname + "_sliding"
        # itopdownpd[cname_sliding] = itopdownpd[cname].rolling(window=5, center=True, min_periods=1).agg("max").astype(
        #     "int")
        # ddrc_rd_mean = itopdownpd[cname_sliding][0:3].mean()  # 得到一个正常值 todo
        # print("{}平均值：{}".format(cname, ddrc_rd_mean))
        # itopdownpd[cname_sliding + "_recover"] = itopdownpd[cname_sliding] + ddrc_rd_mean * mflops_change
        # itopdownpd[cname_sliding + "_recover_sliding"] = itopdownpd[cname_sliding + "_recover"].rolling(window=5, center=True, min_periods=1).agg("max").astype("int")
        #
        # itopdownpd["ddrc_ddwr_sum"] = itopdownpd["ddrc_rd_sliding_recover_sliding"] + itopdownpd[
        #     "ddrc_wr_sliding_recover_sliding"]
    rlistpds = []
    for ipd in predicttopdwnpds:
        tpd = proceeOneTopdownPd(ipd)
        rlistpds.append(tpd)
    return rlistpds

"""
根据mflops的数值，将较低强度的mflops的数值删除点
原始数据不会修改
并且将 连续删除的数据再多删除2个点
"""
def removeUselessDataFromTopdownList(predicttopdownpds: List[pd.DataFrame]) -> List[pd.DataFrame]:
    # 传进来的为5个，如果不为5个，那么默认不舍弃 以中间为标准
    def addOtherPoint(x: pd.Series):
        if False in x.to_list():
            return False
        return True

    def removepdFromtopdown(itopdownpd: pd.DataFrame):
        # mflopsdelete = itopdownpd["mflops"].
        # 先平滑，再删除
        cname = "mflops"
        mflopsdelete = itopdownpd[cname].rolling(window=5, center=True, min_periods=1).median()  # 先取中位数将单个最大点去掉
        mflopsdelete = mflopsdelete.rolling(window=5, center=True, min_periods=1).mean()  # 取平均值中和一下
        isflags = mflopsdelete.apply(lambda x: x > 2000)
        isflags = isflags.rolling(window=5, min_periods=1, center=True).apply(lambda x: addOtherPoint(x)).astype("bool")
        return itopdownpd[isflags].reset_index(drop=True, inplace=False)
    respds = []
    for ipd in predicttopdownpds:
        len1 = len(ipd)
        ipd = removepdFromtopdown(ipd)
        print("因为mflops删除数据:{} 行".format(len1 - len(ipd)))
        respds.append(ipd)
    return respds


"""
处理server 但是需要用到topdown的数据topdown的结合数据

会对原始数值进行处理
"""


def processServerList(predictserverpds: List[pd.DataFrame], predicttopdownpds: List[pd.DataFrame], predictprocesspds ,detectionJson: Dict) -> List[
    pd.DataFrame]:
    alltopdownspd = mergeDataFrames(predicttopdownpds)

    def getSameTime(servertimes: List[str], topdowntimes: List[str]) -> List[str]:
        sametimes = sorted(list(set(servertimes) & set(topdowntimes)))
        return sametimes

    def getsametimepd(servertimepd: pd.DataFrame, alltopdownspd: pd.DataFrame) -> Tuple[Any, Any]:
        sametimes = getSameTime(servertimepd[TIME_COLUMN_NAME].tolist(), alltopdownspd[TIME_COLUMN_NAME].tolist())
        serverchooseindex = servertimepd[TIME_COLUMN_NAME].apply(lambda x: x in sametimes)
        topdownchooseindex = alltopdownspd[TIME_COLUMN_NAME].apply(lambda x: x in sametimes)
        # return datapd[chooseindex][featuresnames].mean()
        return servertimepd[serverchooseindex].reset_index(drop=True), alltopdownspd[topdownchooseindex].reset_index(drop=True)

    def dealServerpdAndTopdownpd(iserverpd: pd.DataFrame,predictprocesspds: List[pd.DataFrame], itopdowndpd: pd.DataFrame, detectionJson: Dict) -> pd.DataFrame:
        assert len(iserverpd) == len(itopdowndpd)
        # 对itopdownpd中的mflops进行平滑处理
        cname = "mflops"
        itopdowndpd[cname] = itopdowndpd[cname].rolling(window=5, center=True, min_periods=1).median() # 先将最大最小值去除
        itopdowndpd[cname] = itopdowndpd[cname].rolling(window=5, center=True, min_periods=1).mean()
        mflops_mean = getNormalTopdownMean(detectionJson, [itopdowndpd], [cname], datanumber=10)[cname]
        mflops_change = itopdowndpd[cname].apply(lambda x: (mflops_mean - x) / mflops_mean if x < mflops_mean else 0)

        cname = "pgfree"
        iserverpd[cname] = iserverpd[cname].rolling(window=5, center=True, min_periods=1).median() # 先将最大最小值去除
        iserverpd[cname] = iserverpd[cname].rolling(window=5, center=True, min_periods=1).median() # 多去一次
        iserverpd[cname] = iserverpd[cname].rolling(window=5, center=True, min_periods=1).mean()
        pgfree_mean = getNormalServerMean(detectionJson, [iserverpd], predictprocesspds, [cname], datanumber=10)[cname]
        iserverpd[cname] = iserverpd[cname] + pgfree_mean * mflops_change
        iserverpd[cname] = iserverpd[cname].rolling(window=5, center=True, min_periods=1).median() # 对pgfree得到的结果重新去掉最大值最小值
        # pgfree 需要减去平均值
        iserverpd[cname] = iserverpd[cname] - pgfree_mean
        return iserverpd


    resserverpds = []
    for iserverpd in predictserverpds:
        spd, tpd = getsametimepd(iserverpd, alltopdownspd)
        ispd = dealServerpdAndTopdownpd(spd,predictprocesspds,tpd, detectionJson=detectionJson)
        resserverpds.append(ispd)
    return resserverpds


"""
将数据进行特征提取

如果requestData为None，那么数据将从inputDict中的predictdirjsonpath字段找到我们的数据来源
如果requestData不为空，那么将直接使用requestData中的来源
"""


def FeatureextractionData(inputDict: Dict, requestData: Dict = None):
    print("将数据从文件中读取".center(40, "*"))
    detectionJson = getDetectionJsonFrominputconfig(inputDict, requestData, isChangeInfo=False)

    predictserverpds = getServerPdFromJsonDict(sdict=detectionJson)
    predictprocesspds = getProcessPdFromJsonDict(sdict=detectionJson)
    predictl2pds = getL2PdFromJsonDict(sdict=detectionJson)
    predictnetworkpds = getNetworkPdFromJsonDict(sdict=detectionJson)
    predictpingpds = getPingPdFromJsonDict(sdict=detectionJson)
    predicttopdwnpds = getTopdownPdFromJsonDict(sdict=detectionJson)

    if not inputDict["isTimeisStr"]:
        print("对时间由数字转化为字符串".format(40, "*"))
        changeTimePdsToStrFromInt(predictserverpds)
        changeTimePdsToStrFromInt(predictprocesspds)
        changeTimePdsToStrFromInt(predictl2pds)
        changeTimePdsToStrFromInt(predictnetworkpds)
        changeTimePdsToStrFromInt(predictpingpds)
        changeTimePdsToStrFromInt(predicttopdwnpds)

    print("将数据的时间进行统一化处理".center(40, "*"))
    predictserverpds = changeTimeToFromPdlists(predictserverpds, isremoveDuplicate=True)
    predictprocesspds = changeTimeToFromPdlists(predictprocesspds)
    predictl2pds = changeTimeToFromPdlists(predictl2pds, isremoveDuplicate=True)
    predictnetworkpds = changeTimeToFromPdlists(predictnetworkpds, isremoveDuplicate=True)
    predictpingpds = changeTimeToFromPdlists(predictpingpds, isremoveDuplicate=False)
    predicttopdwnpds = changeTimeToFromPdlists(predicttopdwnpds, isremoveDuplicate=True)

    print("对读取到的原始数据进行差分".format(40, "*"))
    # 对异常数据进行差分处理之后，得到cpu特征值 process数据会去掉前三min和后三min的数值
    predictprocesspds = differenceProcess(predictprocesspds, inputDict["process_accumulate_feature"])
    # 对异常server服务数据进行差分处理之后，得到一些指标
    predictserverpds = differenceServer(predictserverpds, inputDict["server_accumulate_feature"])
    predictnetworkpds = differenceServer(predictnetworkpds, inputDict["network_accumulate_feature"])
    predictl2pds = differenceServer(predictl2pds, inputDict["l2_accumulate_feature"])
    predictpingpds = differenceServer(predictpingpds, inputDict["ping_accumulate_feature"])
    predicttopdwnpds = differenceServer(predicttopdwnpds, inputDict["topdown_accumulate_feature"])

    print("根据topdown数据对数据进行对齐操作".format(40, "*"))
    # 对mflops分析然后是删除掉不够的部分
    predicttopdwnpds = removeUselessDataFromTopdownList(predicttopdwnpds)
    # 将时间与对应位置对齐
    predictserverpds = getRunHPCTimepdsFromProcess(predictserverpds, predicttopdwnpds)
    predictprocesspds = getRunHPCTimepdsFromProcess(predictprocesspds, predicttopdwnpds)

    # ============================================================ 对数据进行修改
    # 1. 对inputDict中的特征进行修改  保证下面对其进行标准化
    inputDict["process_feature"] = ["cpu"]  # cpu使用的特征值变为cpu
    inputDict["topdown_feature"] = [] # 原因是不需要对任何指标进行特征提取额

    # 2. 对topdown原始数据数据进行处理 对读写数据进行补偿性操作
    predicttopdwnpds = processTopdownList(detectionJson, predicttopdwnpds)

    # 3. 对process数据进行处理
    add_cpu_column(predictprocesspds)

    # 4. 对server数据进行处理 需要对server中进行补偿性处理,所以需要topdown数据
    # 补偿之后对pgfree进行减去平均值，使其与2M进行判断
    predictserverpds = processServerList(predictserverpds, predicttopdwnpds,predictprocesspds, detectionJson)

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
    normaltopdown_meanvalue = getNormalTopdownMean(detectionJson, predicttopdwnpds, inputDict["topdown_feature"],
                                                   datanumber=inputDict["meanNormalDataNumber"])
    # ---- 不对ping求平均值

    # 将数据进行保存
    if inputDict["spath"] is not None:
        tpath = os.path.join(inputDict["spath"], "1.正常数据的平均值")
        savepdfile(normalserver_meanvalue, tpath, "meanvalue_server.csv", index=True)
        savepdfile(normalprocess_meanvalue, tpath, "meanvalue_process.csv", index=True)
        savepdfile(normall2_meanvalue, tpath, "meanvalue_l2.csv", index=True)
        savepdfile(normalnetwork_meanvalue, tpath, "meanvalue_network.csv", index=True)
        savepdfile(normaltopdown_meanvalue, tpath, "meanvalue_topdown.csv", index=True)
    # ========================================

    print("标准化要预测的process和server数据".center(40, "*"))
    standardserverfeatures = removeListValues(inputDict["server_feature"], ["pgfree"]) # pgfree的判断不在于标准化
    standard_server_pds = standardLists(pds=predictserverpds, standardFeatures=standardserverfeatures, meanValue=normalserver_meanvalue, standardValue=100)
    standard_process_pds = standardLists(pds=predictprocesspds, standardFeatures=inputDict["process_feature"],
                                         meanValue=normalprocess_meanvalue, standardValue=60)
    standard_l2_pds = standardLists(pds=predictl2pds, standardFeatures=inputDict["l2_feature"],
                                    meanValue=normall2_meanvalue, standardValue=100)
    standard_network_pds = standardLists(pds=predictnetworkpds, standardFeatures=inputDict["network_feature"],
                                         meanValue=normalnetwork_meanvalue, standardValue=100)
    standard_topdown_pds = standardLists(pds=predicttopdwnpds, standardFeatures=inputDict["topdown_feature"],
                                         meanValue=normaltopdown_meanvalue, standardValue=100)
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
        saveDFListToFiles(os.path.join(tpath, "topdown_standard"), standard_topdown_pds)

    # print("将数据的时间进行统一化处理".center(40, "*"))
    # standard_server_pds = changeTimeToFromPdlists(standard_server_pds, isremoveDuplicate=True)
    # standard_process_pds = changeTimeToFromPdlists(standard_process_pds)
    # standard_l2_pds = changeTimeToFromPdlists(standard_l2_pds, isremoveDuplicate=True)
    # standard_network_pds = changeTimeToFromPdlists(standard_network_pds, isremoveDuplicate=True)
    # standard_ping_pds = changeTimeToFromPdlists(standard_ping_pds, isremoveDuplicate=False)
    # standard_topdown_pds = changeTimeToFromPdlists(standard_topdown_pds, isremoveDuplicate=False)

    print("process、server、l2、network特征处理".center(40, "*"))
    extraction_process_pds = extractionProcessPdLists(standard_process_pds,
                                                      extractFeatures=inputDict["process_feature"],
                                                      windowsSize=3)
    extraction_server_pds = extractionServerPdLists(standard_server_pds, extractFeatures=inputDict["server_feature"],
                                                    windowsSize=3)
    extraction_l2_pds = extractionServerPdLists(standard_l2_pds, extractFeatures=inputDict["l2_feature"], windowsSize=3)
    extraction_network_pds = extractionServerPdLists(standard_network_pds, extractFeatures=inputDict["network_feature"],
                                                     windowsSize=3)
    extraction_topdown_pds = extractionServerPdLists(standard_topdown_pds, extractFeatures=inputDict["topdown_feature"],
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
        saveDFListToFiles(os.path.join(tpath, "topdown_extraction"), extraction_topdown_pds)
    return extraction_server_pds, extraction_process_pds, extraction_l2_pds, extraction_network_pds, extraction_ping_pds, extraction_topdown_pds


"""
返回一个L2L3层合并之后的数据
"""


def detectionL2L3Data(inputDict: Dict, allserverpds: pd.DataFrame, allprocesspds: pd.DataFrame,
                      alll2pds: pd.DataFrame, allnetworkpds: pd.DataFrame, allpingpds: pd.DataFrame,
                      alltopdownpds: pd.DataFrame) -> pd.DataFrame:
    # 需要用到的特征值
    resfeatures = [TIME_COLUMN_NAME, FAULT_FLAG, "preFlag"]

    print("对L3层CPU异常进行预测".center(40, "*"))
    tpath = None
    if inputDict["spath"] is not None:
        tpath = os.path.join(inputDict["spath"], "4.CPU异常检测中间文件")
    l3cpuresult = pd.DataFrame()
    l3cpuresult[TIME_COLUMN_NAME] = allserverpds[TIME_COLUMN_NAME]
    if inputDict["isExistFaultFlag"]:
        l3cpuresult[FAULT_FLAG] = allserverpds[FAULT_FLAG]
    l3cpuresult["preFlag"] = detectL3CPUAbnormal(allserverpds=allserverpds, allprocesspds=allprocesspds, spath=tpath,
                                                 modelfilepath=inputDict["processcpu_modelpath"],
                                                 modeltype=inputDict["processcpu_modeltype"])

    l3_server_topdownpds = mergeinnerTwoDataFrame(lpd=allserverpds, rpd=alltopdownpds)  # 根据时间得到server和topdown的合并结果

    print("对L3层内存泄露进行检测".center(40, "*"))
    l3memleakresult = pd.DataFrame()
    l3memleakresult[TIME_COLUMN_NAME] = l3_server_topdownpds[TIME_COLUMN_NAME]
    if inputDict["isExistFaultFlag"]:
        l3memleakresult[FAULT_FLAG] = l3_server_topdownpds[FAULT_FLAG]
    l3memleakresult["preFlag"] = detectL3MemLeakAbnormal(allserverpds=l3_server_topdownpds,
                                                         modelfilepath=inputDict["servermemory_modelpath"],
                                                         modeltype=inputDict["servermemory_modeltype"])

    print("对L3层内存带宽进行检测".center(40, "*"))
    l3BandWidthResult = pd.DataFrame()
    l3BandWidthResult[TIME_COLUMN_NAME] = l3_server_topdownpds[TIME_COLUMN_NAME]
    if inputDict["isExistFaultFlag"]:
        l3BandWidthResult[FAULT_FLAG] = l3_server_topdownpds[FAULT_FLAG]
    l3BandWidthResult["preFlag"] = detectL3BandWidthAbnormal(allserverpds=l3_server_topdownpds,
                                                             modelfilepath=inputDict["serverbandwidth_modelpath"],
                                                             modeltype=inputDict["serverbandwidth_modeltype"])
    print("对cache抢占进行检测".center(40, "*"))
    l3CacheGrabResult = pd.DataFrame()
    l3CacheGrabResult[TIME_COLUMN_NAME] = l3_server_topdownpds[TIME_COLUMN_NAME]
    if inputDict["isExistFaultFlag"]:
        l3CacheGrabResult[FAULT_FLAG] = l3_server_topdownpds[FAULT_FLAG]
    l3CacheGrabResult["preFlag"] = predictCacheGrab(l3_server_topdownpds, l3BandWidthResult, modelfilepath=inputDict["cachegrab_modelpath"], modeltype=inputDict["cachegrab_modeltype"])


    print("对L2层数据进行预测".center(40, "*"))
    l2_serverpds = mergeinnerTwoDataFrame(lpd=alll2pds, rpd=allserverpds)  # 根据时间得到l2的合并结果

    # 1. 先预测温度 131  2. 预测机柜功率封顶 121 3 接着预测  111 机器功率封顶  4. 161

    print("1. 对温度进行预测".center(40, "#"))
    l2temperamentresult = pd.DataFrame()
    l2temperamentresult[TIME_COLUMN_NAME] = l2_serverpds[TIME_COLUMN_NAME]
    if inputDict["isExistFaultFlag"]:
        l2temperamentresult[FAULT_FLAG] = l2_serverpds[FAULT_FLAG]
    # l2temperamentresult["preFlag"] = ThransferRightLabels(select_and_pred(l2_serverpds, MODEL_TYPE[tempertature_modeltype], saved_model_path=temperature_modelpath))
    l2temperamentresult["preFlag"] = ThransferRightLabels(predictTemp(model_path=inputDict["temperature_modelpath"],
                                                                      model_type=MODEL_TYPE[
                                                                          inputDict["tempertature_modeltype"]],
                                                                      data=l2_serverpds))
    print("2. 对L2机柜封顶进行预测".center(40, "#"))
    l2cabinetpowerresult = pd.DataFrame()
    l2cabinetpowerresult[TIME_COLUMN_NAME] = l2_serverpds[TIME_COLUMN_NAME]
    if inputDict["isExistFaultFlag"]:
        l2cabinetpowerresult[FAULT_FLAG] = l2_serverpds[FAULT_FLAG]
    # l2cabinetpowerresult["preFlag"] = ThransferRightLabels(select_and_pred(l2_serverpds, MODEL_TYPE[inputDict["power_cabinet_modeltype"]], saved_model_path=inputDict["power_cabinet_modelpath"]))
    l2cabinetpowerresult["preFlag"] = predictCabinet_PowerCapping(model_path=inputDict["power_cabinet_modelpath"], model_type=MODEL_TYPE[inputDict["power_cabinet_modeltype"]], l2_serverdata=l2_serverpds)


    print("3. 对L2机器封顶进行预测".center(40, "#"))
    l2machinepowerresult = pd.DataFrame()
    l2machinepowerresult[TIME_COLUMN_NAME] = l2_serverpds[TIME_COLUMN_NAME]
    if inputDict["isExistFaultFlag"]:
        l2machinepowerresult[FAULT_FLAG] = l2_serverpds[FAULT_FLAG]
    # l2machinepowerresult["preFlag"] = ThransferRightLabels(select_and_pred(l2_serverpds, MODEL_TYPE[inputDict["power_machine_modeltype"]], saved_model_path=inputDict["power_machine_modelpath"]))
    l2machinepowerresult["preFlag"] = predictServer_PowerCapping(model_path=inputDict["power_machine_modelpath"], model_type=MODEL_TYPE[inputDict["power_machine_modeltype"]], l2_serverdata=l2_serverpds, resultPds=[l2temperamentresult, l2cabinetpowerresult])


    print("4. 对CPU主频下降进行预测".center(40, "#"))
    l2cpudownresult = pd.DataFrame()
    l2cpudownresult[TIME_COLUMN_NAME] = l2_serverpds[TIME_COLUMN_NAME]
    if inputDict["isExistFaultFlag"]:
        l2cpudownresult[FAULT_FLAG] = l2_serverpds[TIME_COLUMN_NAME]
    # l2cpudownresult["preFlag"] = predictL2_CPUDown(l2_serverdata=l2_serverpds, freqDownResultPds=[l2machinepowerresult, l2cabinetpowerresult, l2temperamentresult])
    l2cpudownresult["preFlag"] = predictL2_CPUDown(model_path=inputDict["cpudown_modelpath"], model_type=MODEL_TYPE[inputDict["cpudown_modeltype"]], l2_serverdata=l2_serverpds, resultPds=[l2temperamentresult, l2cabinetpowerresult])


    print("对网络异常1进行预测 TX_Hang".center(40, "#"))
    # REPORT_TIME = "time"
    # l2networkresult1 = pd.DataFrame()
    # l2networkresult1[TIME_COLUMN_NAME] = allnetworkpds[REPORT_TIME]
    # l2networkresult1[FAULT_FLAG] = allnetworkpds[FAULT_FLAG]
    # l2networkresult1 = makeL2networkresultMergedByMin(l2networkresult1)
    l2networkresult1 = detectNetwork_TXHangAbnormal(allpingpds, isExistFlag=inputDict["isExistFaultFlag"])

    print("对网络异常2 pfc进行预测".center(40, "#"))
    l2networkresult2 = pd.DataFrame()
    l2networkresult2[TIME_COLUMN_NAME] = allnetworkpds[TIME_COLUMN_NAME]
    if inputDict["isExistFaultFlag"]:
        l2networkresult2[FAULT_FLAG] = allnetworkpds[FAULT_FLAG]
    l2networkresult2["preFlag"] = ThransferRightLabels(
        select_and_pred(allnetworkpds, MODEL_TYPE[inputDict["network_pfctype"]],
                        saved_model_path=inputDict["network_pfcpath"]))
    l2networkresult2 = makeL2networkresultMergedByMin(l2networkresult2)

    print("将L2 L3 Network数据合并分析".center(40, "*"))
    allresultspd = mergeouterPredictResult(
        [l3cpuresult, l3memleakresult, l3BandWidthResult, l3CacheGrabResult,l2machinepowerresult, l2cabinetpowerresult,
         l2temperamentresult, l2networkresult1, l2networkresult2, l2cpudownresult], isExistFlag=inputDict["isExistFaultFlag"])

    print("对结果进行优化".center(40, "*"))
    if inputDict["isExistFaultFlag"]:
        allresultspd = fixFaultFlag(allresultspd)
    allresultspd = fixIsolatedPointPreFlag(allresultspd)

    if inputDict["isExistFaultFlag"]:
        print("增加此时间点是否预测正确".center(40, "*"))
        if inputDict["isExistFaultFlag"]:
            isrightLists = [1 if allresultspd[FAULT_FLAG][i] != 0 and (allresultspd[FAULT_FLAG][i] in allresultspd["preFlag"][i] or allresultspd[FAULT_FLAG][i] // 10 * 10 in allresultspd["preFlag"][i]) else 0 for i in range(0, len(allresultspd))]
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
    if "resultsavepath" in inputDict and inputDict["resultsavepath"] is not None:  # 路径不能为空，如果为空，默认为当前目录
        resultsavepath = os.path.abspath(inputDict["resultsavepath"])
        # if inputDict["resultsavepath"] == ".":
        #     resultsavepath = sys.path[0]
        # print("绝对路径：{}".format(resultsavepath))
        if not os.path.exists(resultsavepath):
            print("输出结果路径:{}不存在".format(resultsavepath))
            exit(1)
    else:
        return  # 不进行保存
    if "outputJsonFilename" in inputDict and inputDict["outputJsonFilename"] is not None:
        outputJsonFilename = inputDict["outputJsonFilename"]
    saveDictToJson(outputJsonDict, resultsavepath, outputJsonFilename)


"""
从输入json文件中读取信息，进行检测

如果requestData为None，那么数据将从inputDict中的predictdirjsonpath字段找到我们的数据来源
如果requestData不为空，那么将直接使用requestData中的来源
"""


def detectionFromInputDict(inputDict: Dict, requestData: Dict = None) -> Dict:
    # =====================将数据进行特征提取
    # extraction_server_pds, extraction_process_pds, extraction_l2_pds, extraction_network_pds, extraction_ping_pds, extraction_topdown_pds = FeatureextractionData(inputDict, requestData)
    extractionpds = FeatureextractionData(inputDict, requestData)

    # =====================数据合并
    allserverpds = mergeDataFrames(extractionpds[0])
    allprocesspds = mergeDataFrames(extractionpds[1])
    alll2pds = mergeDataFrames(extractionpds[2])
    allnetworkpds = mergeDataFrames(extractionpds[3])
    allpingpds = mergeDataFrames(extractionpds[4])
    alltopdownpds = mergeDataFrames(extractionpds[5])

    print("对L3 L2层的数据进行预测".center(40, "*"))
    l2l3predetectresultpd = detectionL2L3Data(inputDict, allserverpds, allprocesspds, alll2pds, allnetworkpds,
                                              allpingpds, alltopdownpds)
    tpath = None
    if inputDict["spath"] is not None:
        tpath = os.path.join(inputDict["spath"], "5.准确率结果分析")
    if inputDict["isExistFaultFlag"]:
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


"""
对输入的配置文件进行修改
"""
def getDetectionJsonFrominputconfig(inputDict: Dict, requestData = None, isChangeInfo = True) -> Dict:
    if requestData is None:
        requestData = readJsonToDict(*(os.path.split(inputDict["predictdirjsonpath"])))
    # 修改内存带宽中的模型路径
    if not isChangeInfo:
        return requestData
    return requestData

def removeListValues(rlist: List, rvalue: List) -> List:
    resList = []
    for i in rlist:
        if i in rvalue:
            continue
        resList.append(i)
    return resList


