import os

import pandas as pd

from Classifiers.TrainToTest import ModelTrainAndTest, change_threshold
from utils.DataFrameOperation import mergeDataFrames, changePDfaultFlag
from utils.DataScripts import getDFmean, getFaultDataFrame
from utils.DefineData import TIME_COLUMN_NAME, PID_FEATURE, CPU_FEATURE, FAULT_FLAG
from utils.FileSaveRead import saveDFListToFiles
from utils.auto_forecast import getfilespath, getfilepd, differenceProcess, add_cpu_column, differenceServer, \
    standardLists, changeTimeTo_pdlists, processpdsList, serverpdsList, removeAllHeadTail, removeProcessAllHeadTail

if __name__ == "__main__":
    # ============================================================================================= 输入数据定义
    # 先将所有的server文件和process文件进行指定
    # 其中单个server文件我默认是连续的
    predictdirpath = R"DATA\正常和异常数据\训练数据-E5-3km-异常数据"
    predictserverfiles = getfilespath(os.path.join(predictdirpath, "server"))
    predictprocessfiles = getfilespath(os.path.join(predictdirpath, "process"))
    # 指定正常server和process文件路径
    normaldirpath = R"DATA\正常和异常数据\E5-3km-正常数据"
    normalserverfiles = getfilespath(os.path.join(normaldirpath, "server"))
    normalprocessfiles = getfilespath(os.path.join(normaldirpath, "process"))
    # 预测CPU的模型路径 保存的路径
    processcpu_modelpath = R"tmp/modelpath1/singlefeature/process_cpu_model"
    # 预测内存泄露的模型路径
    servermemory_modelpath = R"tmp/modelpath1/singlefeature/memory_leak_model"
    # 预测内存带宽的模型路径
    serverbandwidth_modelpath = R"tmp/modelpath1/singlefeature/memory_bandwidth_model"
    # 将一些需要保存的临时信息进行保存路径
    spath = "tmp/模型训练中间数据/1.训练三种模型"

    # 需要对server数据进行处理的指标
    server_feature = ["used", "pgfree"]
    server_accumulate_feature = ["pgfree"]
    # 需要对process数据进行处理的指标, cpu数据要在数据部分添加, 在后面，会往这个列表中添加一个cpu数据
    process_feature = ["user", "system"]
    process_accumulate_feature = ["user", "system"]

    # 在处理时间格式的时候使用，都被转化为'%Y-%m-%d %H:%M:00' 在这里默认所有的进程数据是同一种时间格式，
    server_time_format = '%Y/%m/%d %H:%M'
    process_time_format = '%Y/%m/%d %H:%M'

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
    isnormalDataFromNormal = True # 判断是否使用来自normal数据中的  必须保证有正常数据
    # 训练模型的指标
    maxdepth = 5
    model_memLeak_features = ["used_mean"] # 训练内存泄漏模型需要的指标
    model_memBandwidth_features = ["pgfree_mean"]
    model_cpu_features = ["cpu", "cpu_mean"]



    # 如果使用了手工指定的平均值，那么正常数据的来源必须是异常数据
    if not isFileMean:
        isnormalDataFromNormal = False # 如果平均值不来自文件 代表没有正常值 那么将会使用默认使用异常文件中的正常数据

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
    time_server_feature.extend([TIME_COLUMN_NAME, FAULT_FLAG])
    time_process_feature.extend([TIME_COLUMN_NAME, PID_FEATURE, CPU_FEATURE, FAULT_FLAG])

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
    # 如果normal数据存在，也要对normal数据进行标准化
    if isFileMean:
        standard_normal_server_pds = standardLists(pds=normalserverpds, standardFeatures=server_feature,
                                            meanValue=normalserver_meanvalue, standardValue=100)
        standard_normal_process_pds = standardLists(pds=normalprocesspds, standardFeatures=process_feature,
                                             meanValue=normalprocess_meanvalue, standardValue=60)

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
    # 如果normal数据存在 也要进行处理
    if isFileMean:
        standard_normal_server_pds = changeTimeTo_pdlists(standard_normal_server_pds, server_time_format)
        standard_normal_process_pds = changeTimeTo_pdlists(standard_normal_process_pds, process_time_format)

    standard_server_pds = changeTimeTo_pdlists(standard_server_pds, server_time_format)
    standard_process_pds = changeTimeTo_pdlists(standard_process_pds, process_time_format)
    # ============================================================================================= 对process数据和server数据进行特征提取
    # 如果normal数据存在也要对normal数据进行特征处理
    if isFileMean:
        print("对正常数据process数据进行特征处理".center(40, "*"))
        tpath = os.path.join(spath, "3.1 正常数据process特征提取数据")
        # 将cpu特征添加到process_feature中 accumulateFeatures是个没用的指标
        extraction_normal_process_pds = processpdsList(standard_normal_process_pds, extractFeatures=process_feature,
                                                accumulateFeatures=process_feature, windowsSize=3, spath=tpath)
        print("对正常数据server数据进行特征处理".center(40, "*"))
        tpath = os.path.join(spath, "4.1 正常数据server特征提取数据")
        extraction_normal_server_pds = serverpdsList(standard_normal_server_pds, extractFeatures=server_feature,
                                              windowsSize=3, spath=tpath)

    print("对异常数据process数据进行特征处理".center(40, "*"))
    tpath = os.path.join(spath, "3.2 异常数据process特征提取数据")
    # 将cpu特征添加到process_feature中 accumulateFeatures是个没用的指标
    extraction_process_pds = processpdsList(standard_process_pds, extractFeatures=process_feature,
                                            accumulateFeatures=process_feature, windowsSize=3, spath=tpath)
    print("对异常数据server数据进行特征处理".center(40, "*"))
    tpath = os.path.join(spath, "4.2 异常数据server特征提取数据")
    extraction_server_pds = serverpdsList(standard_server_pds, extractFeatures=server_feature,
                                          windowsSize=3, spath=tpath)

    # ============================================================================================= 训练内存泄漏模型和内存带宽模型
    # 正常的数据处理
    allserverpds, _ = mergeDataFrames(extraction_server_pds)
    # 对server数据中异常开始前后的是某些数据进行剔除
    alldealedserverpds = removeAllHeadTail(allserverpds)

    # 如果isnormalDataFromNormal=True 得到正常数据中的正常数据 否则得到异常数据中的数据
    if isnormalDataFromNormal:
        normalTrainData, _ = mergeDataFrames(extraction_normal_server_pds)
    else:
        normalTrainData = getFaultDataFrame(alldealedserverpds, [0])
    # ------
    print("训练内存泄露模型".center(40, "*"))
    tpath = os.path.join(spath, "5. 训练内存泄露模型中间数据")
    # 得到异常数据中的内存泄露数据used指标
    allabnormalTrainData = getFaultDataFrame(alldealedserverpds, [61,62,63,64,65])
    abnormalTrainData = changePDfaultFlag(allabnormalTrainData)
    allTrainedPD,_ = mergeDataFrames([normalTrainData, abnormalTrainData])
    ModelTrainAndTest(allTrainedPD, None,testAgain=False, spath=tpath, selectedFeature=model_memLeak_features,
                      modelpath=servermemory_modelpath, maxdepth=maxdepth)
    # change_threshold(os.path.join(servermemory_modelpath, "decision_tree.pkl"), 0, 56.8)
    # 将训练的正常数据和异常数据进行保存
    normalTrainData.to_csv(os.path.join(tpath, "0.正常训练数据.csv"))
    allabnormalTrainData.to_csv(os.path.join(tpath, "0.异常训练数据.csv"))
    allTrainedPD.to_csv(os.path.join(tpath, "0.正常异常合并训练数据.csv"))


    print("训练内存带宽模型".center(40, "*"))
    tpath = os.path.join(spath, "6. 训练内存带宽模型中间数据")
    allabnormalTrainData = getFaultDataFrame(alldealedserverpds, [52, 53, 54, 55])
    abnormalTrainData = changePDfaultFlag(allabnormalTrainData)
    allTrainedPD, _ = mergeDataFrames([normalTrainData, abnormalTrainData])
    ModelTrainAndTest(allTrainedPD, None, testAgain=False, spath=tpath,
                      selectedFeature=model_memBandwidth_features,
                      modelpath=serverbandwidth_modelpath, maxdepth=maxdepth)
    # 将训练的正常数据和异常数据进行保存
    normalTrainData.to_csv(os.path.join(tpath, "0.正常训练数据.csv"))
    allabnormalTrainData.to_csv(os.path.join(tpath, "0.异常训练数据.csv"))
    allTrainedPD.to_csv(os.path.join(tpath, "0.正常异常合并训练数据.csv"))
    change_threshold(os.path.join(servermemory_modelpath, "decision_tree.pkl"), 0, 120)

    # ============================================================================================= 训练内存泄漏模型和内存带宽模型
    # 正常的数据处理
    print("训练CPU异常模型".center(40, "*"))
    tpath = os.path.join(spath, "7. 训练CPU异常模型中间数据")
    allprocesspds, _ = mergeDataFrames(extraction_process_pds)
    allprocesspds = removeProcessAllHeadTail(allprocesspds, windowsize=3) # 出去异常前后两个点
    if isnormalDataFromNormal:
        normalTrainData, _ = mergeDataFrames(extraction_normal_process_pds)
    else:
        normalTrainData = getFaultDataFrame(allprocesspds, [0])

    allabnormalTrainData = getFaultDataFrame(allprocesspds, [11, 12, 13, 14, 15])
    abnormalTrainData = changePDfaultFlag(allabnormalTrainData)
    allTrainedPD,_ = mergeDataFrames([normalTrainData, abnormalTrainData])
    ModelTrainAndTest(allTrainedPD, None, testAgain=False, spath=tpath, selectedFeature=model_cpu_features,
                      modelpath=processcpu_modelpath, maxdepth=maxdepth)
    change_threshold(os.path.join(processcpu_modelpath, "decision_tree.pkl"), 0, 56.8)
    normalTrainData.to_csv(os.path.join(tpath, "0.正常训练数据.csv"))
    allabnormalTrainData.to_csv(os.path.join(tpath, "0.异常训练数据.csv"))
    allTrainedPD.to_csv(os.path.join(tpath, "0.正常异常合并训练数据.csv"))


