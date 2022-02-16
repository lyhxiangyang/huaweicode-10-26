import os
from typing import Set, Tuple

import pandas as pd

from utils.DataFrameOperation import mergeDataFrames
from utils.DataScripts import getDFmean
from utils.DefineData import TIME_COLUMN_NAME, PID_FEATURE, CPU_FEATURE, FAULT_FLAG
from utils.FileSaveRead import saveDFListToFiles
from utils.auto_forecast import getfilespath, getfilepd, differenceProcess, add_cpu_column, differenceServer, \
    standardLists, changeTimeTo_pdlists, processpdsList, serverpdsList, deal_serverpds_and_processpds, \
    predictAllAbnormal, analysePredictResult

if __name__ == "__main__":
    # ============================================================================================= 输入数据定义
    # 先将所有的server文件和process文件进行指定
    # 其中单个server文件我默认是连续的
    predictdirpath = R"DATA\正常和异常数据\训练数据-E5-9km-异常数据"
    predictserverfiles = getfilespath(os.path.join(predictdirpath, "server"))
    predictprocessfiles = getfilespath(os.path.join(predictdirpath, "process"))
    # 指定正常server和process文件路径
    normaldirpath = R"DATA\正常和异常数据\E5-3km-正常数据"
    normalserverfiles = getfilespath(os.path.join(normaldirpath, "server"))
    normalprocessfiles = getfilespath(os.path.join(normaldirpath, "process"))
    # 预测CPU的模型路径
    processcpu_modelpath = R"hpc/models/l3/memory_bandwidth_model"
    # 预测内存泄露的模型路径
    servermemory_modelpath = R"hpc/models/l3/memory_leak_model"
    # 预测内存带宽的模型路径
    serverbandwidth_modelpath = R"hpc/models/l3/memory_bandwidth_model"
    # 将一些需要保存的临时信息进行保存路径
    spath = "tmp/总过程分析/训练数据-E5-9km"
    # 是否有存在faultFlag
    isExistFaultFlag = True
    # 核心数据 如果isManuallyspecifyCoreList==True那么就专门使用我手工指定的数据，如果==False，那么我使用的数据就是从process文件中推理出来的结果
    coresnumber = 104  # 运行操作系统的实际核心数  如实填写
    isManuallyspecifyCoreList = False
    wrfruncoresnumber = 104  # wrf实际运行在的核心数，如果isManuallyspecifyCoreList = False将会手工推导演
    coresSet = set(range(0, 103))  # wrf实际运行在的核心数

    # 需要对server数据进行处理的指标
    server_feature = ["used", "pgfree"]
    server_accumulate_feature = ["pgfree"]
    # 需要对process数据进行处理的指标, cpu数据要在数据部分添加, 在后面，会往这个列表中添加一个cpu数据
    process_feature = ["user", "system"]
    process_accumulate_feature = ["user", "system"]

    # 在处理时间格式的时候使用，都被转化为'%Y-%m-%d %H:%M:00' 在这里默认所有的进程数据是同一种时间格式，
    server_time_format = '%Y/%m/%d %H:%M'
    process_time_format = '%Y/%m/%d %H:%M'

    # 预测是否使用阀值, True为使用阀值预测 必须指定thresholdValueDict， False代表使用模型进行预测, 必须设置好模型的路径
    isThreshold = True
    thresholdValueDict = {
        "process_cpu_mean": 57,
        "used": 120,  # 不要改key值
        "pgfree": 500
    }
    # 是否使用正常文件中的平均值 True代表这个从正常文件中读取，False代表着直接从字典中读取
    isFileMean = True
    # 如果上面的是False，则使用下面的字典数据
    processmeanVaule = {
        "cpu": 60,
    }
    servermeanValue = {
        "used": 0,
        "pgfree": 0
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

    # 如果为True 才能保证有normal数据
    if isFileMean:
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
    # ============================================================================================= 对读取到的数据进行差分，并且将cpu添加到要提取的特征中
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
        # 得到正常数据的平均值
        normalserver_meanvalue = getDFmean(allnormalserverpd, server_feature)
        normalprocess_meanvalue = getDFmean(allnormalprocesspd, process_feature)
    else:
        normalserver_meanvalue = pd.Series(data=servermeanValue)
        normalprocess_meanvalue = pd.Series(data=processmeanVaule)
    # 将这几个平均值进行保存
    tpath = os.path.join(spath, "1. 正常数据的平均值")
    if not os.path.exists(tpath):
        os.makedirs(tpath)
    normalprocess_meanvalue.to_csv(os.path.join(tpath, "meanvalue_process.csv"))
    normalserver_meanvalue.to_csv(os.path.join(tpath, "meanvalue_server.csv"))

    # ============================================================================================= 对要预测的数据进行标准化处理
    # 标准化process 和 server数据， 对于process数据，先将cpu想加在一起，然后在求平均值。
    print("标准化要预测的process和server数据".center(40, "*"))
    standard_server_pds = standardLists(pds=predictserverpds, standardFeatures=server_feature,
                                        meanValue=normalserver_meanvalue, standardValue=100)
    standard_process_pds = standardLists(pds=predictprocesspds, standardFeatures=process_feature,
                                         meanValue=normalprocess_meanvalue, standardValue=60)

    # 对标准化结果进行存储
    tpath = os.path.join(spath, "2. 标准化数据存储")
    saveDFListToFiles(os.path.join(tpath, "server_standard"), standard_server_pds)
    saveDFListToFiles(os.path.join(tpath, "process_standard"), standard_process_pds)
    # ============================================================================================= 对process数据和server数据进行秒数的处理，将秒数去掉
    standard_server_pds = changeTimeTo_pdlists(standard_server_pds, server_time_format)
    standard_process_pds = changeTimeTo_pdlists(standard_process_pds, process_time_format)
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

    # ============================================================================================= 将process数据和server数据合在一起，按照server时间进行预测
    print("将提取之后的server数据和process数据进行合并".center(40, "*"))
    tpath = os.path.join(spath, "5. server和process合并")
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
        writeinfo = ["系统核心数量{}\n".format(coresnumber), "核心数量：{}\n".format(wrfruncoresnumber),
                     "核心的位数：{}\n".format(coresSet)]
        f.writelines(writeinfo)

    # ============================================================================================= 对process数据和server数据合在一起进行预测
    # 只有存在FaultFlag才能进行预测
    if isExistFaultFlag:
        print("对server数据和process数据进行预测".center(40, "*"))
        tpath = os.path.join(spath, "6. 最终预测结果")
        # time  faultFlag  preFlag  mem_leak  mem_bandwidth
        predictpd = predictAllAbnormal(
            serverinformationDict=serverinformationDict,
            spath=tpath,
            isThreshold=isThreshold,
            thresholdValue=thresholdValueDict,
            Memory_bandwidth_modelpath=serverbandwidth_modelpath,
            Memory_leaks_modelpath=servermemory_modelpath,
            coresnumber=wrfruncoresnumber
        )
        # 对结果进行分析
        analysePredictResult(predictpd, tpath)

