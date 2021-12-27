import os
from typing import Set, Tuple, List, Any

import pandas as pd

from Classifiers.ModelPred import select_and_pred
from utils.DataFrameOperation import mergeDataFrames, mergeinnerTwoDataFrame, mergeouterPredictResult
from utils.DataScripts import getDFmean
from utils.DefineData import TIME_COLUMN_NAME, PID_FEATURE, CPU_FEATURE, FAULT_FLAG, MODEL_TYPE
from utils.FileSaveRead import saveDFListToFiles
from utils.auto_forecast import getfilespath, getfilepd, differenceProcess, add_cpu_column, differenceServer, \
    standardLists, changeTimeTo_pdlists, processpdsList, serverpdsList, deal_serverpds_and_processpds, \
    predictAllAbnormal, analysePredictResult

# 得到四种列表
def getServer_Process_l2_NetworkList(
        dirpath: str,
        server_feature,
        process_feature,
        l2_feature,
        network_feature,
        isExistFlag: bool = True,
)-> tuple[list[Any], list[Any], list[Any], list[Any]]:
    serverfiles = getfilespath(os.path.join(dirpath, "server"))
    processfiles = getfilespath(os.path.join(dirpath, "process"))
    l2files = getfilespath(os.path.join(dirpath, "l2"))
    networkfiles = getfilespath(os.path.join(dirpath, "network"))

    time_server_feature = server_feature.copy()
    time_process_feature = process_feature.copy()
    time_l2_feature = l2_feature.copy()
    time_network_feature = network_feature.copy()

    time_server_feature.extend([TIME_COLUMN_NAME])
    time_process_feature.extend([TIME_COLUMN_NAME, PID_FEATURE, CPU_FEATURE])
    time_l2_feature.extend([TIME_COLUMN_NAME])
    time_network_feature.extend(["report_time"])
    if isExistFlag:
        time_server_feature.extend([FAULT_FLAG])
        time_process_feature.extend([FAULT_FLAG])
        time_l2_feature.extend([FAULT_FLAG])
        time_network_feature.extend([FAULT_FLAG])

    processpds = []
    serverpds = []
    l2pds = []
    networkpds = []
    # 预测进程数据
    for ifile in processfiles:
        tpd = getfilepd(ifile)
        tpd = tpd.loc[:, time_process_feature]
        processpds.append(tpd)
    # 预测服务数据
    for ifile in serverfiles:
        tpd = getfilepd(ifile)
        tpd = tpd.loc[:, time_server_feature]
        serverpds.append(tpd)
    # 预测l2数据
    for ifile in l2files:
        tpd = getfilepd(ifile)
        tpd = tpd.loc[:, time_l2_feature]
        l2pds.append(tpd)
    # 预测网络数据
    for ifile in networkfiles:
        tpd = getfilepd(ifile)
        tpd = tpd.loc[:, time_network_feature]
        networkpds.append(tpd)
    return serverpds, processpds, l2pds, networkpds




if __name__ == "__main__":
    # ============================================================================================= 输入数据定义
    # 先将所有的server文件和process文件进行指定
    # 其中单个server文件我默认是连续的
    predictdirpath = R"DATA\L3l2数据集合"
    # 指定正常server和process文件路径
    normaldirpath = R"DATA\L3l2数据集合正常值"
    # 预测CPU的模型路径
    processcpu_modelpath = R"tmp/modelpath/singlefeature/process_cpu_model"
    # 预测内存泄露的模型路径
    servermemory_modelpath = R"tmp/modelpath/singlefeature/memory_leak_model"
    # 预测内存带宽的模型路径
    serverbandwidth_modelpath = R"tmp/modelpath/singlefeature/memory_bandwidth_model"
    # =========下面是l2层需要的模型
    # 机器功率封顶
    power_machine_modelpath = R"tmp/modelpath/l2/机器功率封顶"
    # 机柜功率封顶
    power_cabinet_modelpath = R"tmp/modelpath/l2/机柜功率封顶"
    # 温度过高异常
    temperature_modelpath = R"tmp/modelpath/l2/温度过高"
    # 网络异常
    # 网络异常PFC风暴注入
    network_model1path = R"tmp/modelpath/l2/网络异常1"
    network_model2path = R"tmp/modelpath/l2/网络异常2"
    # 使用哪个模型
    processcpu_modeltype = 0
    servermemory_modeltype = 0
    serverbandwidth_modeltype = 0
    power_machine_modeltype = 0 # 机器功率封顶
    power_cabinet_modeltype = 0 # 机柜功率封顶
    tempertature_modeltype = 0 # 温度过高
    network_model1type = 0 # 网络异常1
    network_model2type = 0 # 网络异常2
    # 将一些需要保存的临时信息进行保存路径
    spath = "tmp/总过程分析/L3l2数据集合"
    # 是否有存在faultFlag
    isExistFaultFlag = True
    # 核心数据 如果isManuallyspecifyCoreList==True那么就专门使用我手工指定的数据，如果==False，那么我使用的数据就是从process文件中推理出来的结果
    coresnumber = 104 # 运行操作系统的实际核心数  如实填写
    isManuallyspecifyCoreList = False
    wrfruncoresnumber = 104 # wrf实际运行在的核心数，如果isManuallyspecifyCoreList = False将会手工推导演
    coresSet = set(range(0, 103)) # wrf实际运行在的核心数

    # 需要对server数据进行处理的指标
    server_feature = ["used", "pgfree", "freq"]
    server_accumulate_feature = ["pgfree"]
    # 需要对process数据进行处理的指标, cpu数据要在数据部分添加, 在后面，会往这个列表中添加一个cpu数据
    process_feature = ["user", "system"]
    process_accumulate_feature = ["user", "system"]
    # 需要对l2数据进行处理的指标，
    l2_feature = ["CPU_Powewr", "Power", "Cabinet_Power", "CPU1_Core_Rem", "FAN1_F_Speed"]
    l2_accumulate_feature = []
    # 需要对网络数据进行处理的指标
    network_feature = ["tx_packets_phy", "rx_packets_phy"]
    network_accumulate_feature = []

    # 在处理时间格式的时候使用，都被转化为'%Y-%m-%d %H:%M:00' 在这里默认所有的进程数据是同一种时间格式，
    # 格式其实是没有用了，函数中已经能够自动分析文件的格式进行使用
    server_time_format = '%Y/%m/%d %H:%M'
    process_time_format = '%Y/%m/%d %H:%M'

    # 预测是否使用阀值, True为使用阀值预测 必须指定thresholdValueDict， False代表使用模型进行预测, 必须设置好模型的路径
    isThreshold = False
    thresholdValueDict = {
        "process_cpu_mean": 57,
        # server的数值
        "used": 120,  # 不要改key值
        "pgfree": 500,
    }
    # 是否使用正常文件中的平均值 True代表这个从正常文件中读取，False代表着直接从字典中读取
    isFileMean = True
    # 如果上面的是False，则使用下面的字典数据
    processmeanVaule = {
        "cpu": 60,
    }
    servermeanValue = {
        "used": 0,
        "pgfree": 0,
        "freq": 0,
    }
    l2meanValue = {
        "CPU_Power": -1,
        "Power": -1,
        "Cabinet_Power": -1,
        "CPU1_Core_Rem": -1,
        "FAN1_F_Speed": -1,
    }
    networkmeanValue = {
        "tx_packets_phy": -1,
        "rx_packets_phy": -1,
    }


    # ============================================================================================= 先将正常数据和预测数据的指标从磁盘中加载到内存中
    print("将数据从文件中读取".center(40, "*"))
    predictserverpds, predictprocesspds, predictl2pds, predictnetworkpds = getServer_Process_l2_NetworkList(
        predictdirpath,
        server_feature,
        process_feature,
        l2_feature,
        network_feature,
    )
    # 如果为True 才能保证有normal数据
    if isFileMean:
        normalserverpds, normalprocesspds, normall2pds, normalnetworkpds = getServer_Process_l2_NetworkList(
            normaldirpath,
            server_feature,
            process_feature,
            l2_feature,
            network_feature
        )

    # ============================================================================================= 对读取到的数据进行差分，并且将cpu添加到要提取的特征中
    # 对l2和network还没有差分的需求
    print("对读取到的原始数据进行差分".format(40, "*"))
    if isFileMean:
        # 对正常进程数据进行差分处理之后，得到cpu特征值
        normalprocesspds = differenceProcess(normalprocesspds, process_accumulate_feature)
        add_cpu_column(normalprocesspds)
        # 对正常server进程数据进行差分处理之后，得到一些指标
        normalserverpds = differenceServer(normalserverpds, server_accumulate_feature)


    # 对异常数据进行差分处理之后，得到cpu特征值
    predictprocesspds = differenceProcess(predictprocesspds, process_accumulate_feature)
    add_cpu_column(predictprocesspds)
    # 对异常server服务数据进行差分处理之后，得到一些指标
    predictserverpds = differenceServer(predictserverpds, server_accumulate_feature)
    # ----
    process_feature = ["cpu"]

    # ============================================================================================= 先对正常数据的各个指标求平均值
    # 往进程指标中只使用"cpu"指标, 需要保证正常数据中的累计值都减去了

    print("先对正常数据的各个指标求平均值".center(40, "*"))
    if isFileMean:
        allnormalserverpd, _ = mergeDataFrames(normalserverpds)
        allnormalprocesspd, _ = mergeDataFrames(normalprocesspds)
        allnormall2pd, _ = mergeDataFrames(normall2pds)
        allnormalNetworkpd, _ = mergeDataFrames(normalnetworkpds)
        # 得到正常数据的平均值
        normalserver_meanvalue = getDFmean(allnormalserverpd, server_feature)
        normalprocess_meanvalue = getDFmean(allnormalprocesspd, process_feature)
        normall2_meanvalue = getDFmean(allnormall2pd, l2_feature)
        normalnetwork_meanvalue = getDFmean(allnormalNetworkpd, network_feature)
    else:
        normalserver_meanvalue = pd.Series(data=servermeanValue)
        normalprocess_meanvalue = pd.Series(data=processmeanVaule)
        normall2_meanvalue = pd.Series(data=l2meanValue)
        normalnetwork_meanvalue = pd.Series(data=networkmeanValue)
    # 将这几个平均值进行保存
    tpath = os.path.join(spath, "1. 正常数据的平均值")
    if not os.path.exists(tpath):
        os.makedirs(tpath)
    normalprocess_meanvalue.to_csv(os.path.join(tpath, "meanvalue_process.csv"))
    normalserver_meanvalue.to_csv(os.path.join(tpath, "meanvalue_server.csv"))
    normall2_meanvalue.to_csv(os.path.join(tpath, "meanvalue_l2.csv"))
    normalnetwork_meanvalue.to_csv(os.path.join(tpath, "meanvalue_network.csv"))

    # ============================================================================================= 对要预测的数据进行标准化处理
    # 标准化process 和 server数据， 对于process数据，先将cpu想加在一起，然后在求平均值。
    print("标准化要预测的process和server数据".center(40, "*"))
    standard_server_pds = standardLists(pds=predictserverpds, standardFeatures=server_feature,
                                        meanValue=normalserver_meanvalue, standardValue=100)
    standard_process_pds = standardLists(pds=predictprocesspds, standardFeatures=process_feature,
                                         meanValue=normalprocess_meanvalue, standardValue=60)
    standard_l2_pds = standardLists(pds=predictl2pds, standardFeatures=l2_feature,
                                    meanValue=normall2_meanvalue, standardValue=100)
    standard_network_pds = standardLists(pds=predictnetworkpds, standardFeatures=network_feature,
                                         meanValue=normalnetwork_meanvalue, standardValue=100)

    # 对标准化结果进行存储
    tpath = os.path.join(spath, "2. 标准化数据存储")
    saveDFListToFiles(os.path.join(tpath, "server_standard"), standard_server_pds)
    saveDFListToFiles(os.path.join(tpath, "process_standard"), standard_process_pds)
    saveDFListToFiles(os.path.join(tpath, "l2_standard"), standard_l2_pds)
    saveDFListToFiles(os.path.join(tpath, "network_standard"), standard_network_pds) #
    # ============================================================================================= 对process数据和server数据进行秒数的处理，将秒数去掉
    standard_server_pds = changeTimeTo_pdlists(standard_server_pds)
    standard_process_pds = changeTimeTo_pdlists(standard_process_pds)
    standard_l2_pds = changeTimeTo_pdlists(standard_l2_pds)
    # standard_network_pds = changeTimeTo_pdlists(standard_network_pds)
    # ============================================================================================= 对process数据和server数据进行特征提取
    print("对process数据进行特征处理".center(40, "*"))
    tpath = os.path.join(spath, "3. process特征提取数据")
    # 将cpu特征添加到process_feature中
    extraction_process_pds = processpdsList(standard_process_pds, extractFeatures=process_feature,
                                            accumulateFeatures=process_feature, windowsSize=3, spath=tpath)
    print("对server数据进行特征处理".center(40, "*"))
    tpath = os.path.join(spath, "4. server特征提取数据")
    extraction_server_pds = serverpdsList(standard_server_pds, extractFeatures=server_feature,
                                          windowsSize=3, spath=tpath)
    tpath = os.path.join(spath, "5. L2特征提取数据")
    extraction_l2_pds = serverpdsList(standard_l2_pds, extractFeatures=l2_feature,
                                          windowsSize=3, spath=tpath)
    tpath = os.path.join(spath, "6. network特征提取")
    extraction_network_pds = serverpdsList(standard_network_pds, extractFeatures=network_feature,
                                           windowsSize=3, spath=tpath)
    # ============================================================================================= 将process数据和server数据合在一起，按照server时间进行预测
    print("将提取之后的server数据和process数据进行合并".center(40, "*"))
    tpath = os.path.join(spath, "7. server和process合并")
    allserverpds, _ = mergeDataFrames(extraction_server_pds)
    allprocesspds, _ = mergeDataFrames(extraction_process_pds)
    # 用于判断L3层异常
    serverinformationDict = deal_serverpds_and_processpds(
        allserverpds=allserverpds,
        allprocesspds=allprocesspds,
        spath=tpath,
        isThreshold=isThreshold,
        thresholdValue=thresholdValueDict,
        modelfilepath=processcpu_modelpath,
        modeltype=processcpu_modeltype,
    )
    # ============================================================================================= 对使用到的核心数进行判断, 因为可能并不是全核心进行预测
    print("使用到的核心数进行判断".center(40, "*"))
    # 得到核心数和核心集合的函数
    def getcores(processpd: pd.DataFrame) -> Tuple[int, Set[int]]:
        coresSet = set(list(processpd[CPU_FEATURE]))
        coresnum = len(coresSet)
        return coresnum, coresSet
    if not isManuallyspecifyCoreList:
        wrfruncoresnumber, coresSet = getcores(allprocesspds)
    print("系统核心数量{}".format(coresnumber))
    print("wrf运行核心数量：{}".format(wrfruncoresnumber))
    print("核心的位数：{}".format(coresSet))
    with open(os.path.join(spath, "运行核心的数据.txt"), "w", encoding="utf-8") as f:
        writeinfo = ["系统核心数量{}\n".format(coresnumber), "核心数量：{}\n".format(wrfruncoresnumber), "核心的位数：{}\n".format(coresSet)]
        f.writelines(writeinfo)

    # ============================================================================================= 对process数据和server数据合在一起进行预测

    if isExistFaultFlag:
        print("对server数据和process数据进行预测".center(40, "*"))
        tpath = os.path.join(spath, "6. 最终预测结果")
        # time  faultFlag  preFlag  mem_leak  mem_bandwidth
        # 对L3进行预测
        predictpd = predictAllAbnormal(
            serverinformationDict=serverinformationDict,
            spath=tpath,
            isThreshold=isThreshold,
            thresholdValue=thresholdValueDict,
            Memory_bandwidth_modelpath=serverbandwidth_modelpath,
            Memory_leaks_modelpath=servermemory_modelpath,
            coresnumber=wrfruncoresnumber,
            memory_leaks_modeltype=servermemory_modeltype,
            memory_bandwidth_modeltype=serverbandwidth_modeltype,
        )
        resfeatures = [TIME_COLUMN_NAME, FAULT_FLAG, "preFlag"]
        L3restult = predictpd[resfeatures]
        # 先得到L2数据
        tpath = os.path.join(spath, "7.L2中间数据")
        if not os.path.exists(tpath):
            os.makedirs(tpath)
        allserverpds.to_csv(os.path.join(tpath, "allserverpd.csv"))
        alll2pds, _ = mergeDataFrames(extraction_l2_pds)
        alll2pds.to_csv(os.path.join(tpath, "alll2pd.csv"))
        l2_serverpds = mergeinnerTwoDataFrame(lpd=alll2pds, rpd=allserverpds) # 根据时间得到l2的合并结果
        # ******* 对L2机器封顶进行预测
        l2machinepowerresult = pd.DataFrame()
        l2machinepowerresult[TIME_COLUMN_NAME] = l2_serverpds[TIME_COLUMN_NAME]
        l2machinepowerresult[FAULT_FLAG] = l2_serverpds[FAULT_FLAG]
        l2machinepowerresult["preFlag"] = select_and_pred(l2_serverpds, MODEL_TYPE[power_machine_modeltype], saved_model_path=power_machine_modelpath)

        # ******* 对L2机柜封顶进行预测
        # 先得到数据
        l2cabinetpowerresult = pd.DataFrame()
        l2cabinetpowerresult[TIME_COLUMN_NAME] = l2_serverpds[TIME_COLUMN_NAME]
        l2cabinetpowerresult[FAULT_FLAG] = l2_serverpds[FAULT_FLAG]
        l2cabinetpowerresult["preFlag"] = select_and_pred(l2_serverpds, MODEL_TYPE[power_cabinet_modeltype], saved_model_path=power_cabinet_modelpath)

        # ******* 对温度进行预测
        l2temperamentresult = pd.DataFrame()
        l2temperamentresult[TIME_COLUMN_NAME] = l2_serverpds[TIME_COLUMN_NAME]
        l2temperamentresult[FAULT_FLAG] = l2_serverpds[FAULT_FLAG]
        l2temperamentresult["preFlag"] = select_and_pred(l2_serverpds, MODEL_TYPE[tempertature_modeltype], saved_model_path=temperature_modelpath)

        # ******* 对网络异常1进行预测
        REPORT_TIME = "report_time"
        allnetworkpds, _ = mergeDataFrames(extraction_network_pds)
        l2networkresult1 = pd.DataFrame()
        l2networkresult1[TIME_COLUMN_NAME] = allnetworkpds[REPORT_TIME]
        l2networkresult1[FAULT_FLAG] = allnetworkpds[FAULT_FLAG]
        l2networkresult1["preFlag"] = select_and_pred(allnetworkpds, MODEL_TYPE[network_model1type], saved_model_path=network_model1path)
        # ******* 对网络异常2进行预测
        l2networkresult2 = pd.DataFrame()
        l2networkresult2[TIME_COLUMN_NAME] = allnetworkpds[REPORT_TIME]
        l2networkresult2[FAULT_FLAG] = allnetworkpds[FAULT_FLAG]
        l2networkresult2["preFlag"] = select_and_pred(allnetworkpds, MODEL_TYPE[network_model2type], saved_model_path=network_model2path)

        # ******* 所有结果分析
        allresults = mergeouterPredictResult([L3restult, l2machinepowerresult, l2cabinetpowerresult, l2temperamentresult, l2networkresult1, l2networkresult2])
        allresults.to_csv(os.path.join(spath, "总体结果.csv"))

        # 对结果进行分析
        # analysePredictResult(allresults, tpath)




