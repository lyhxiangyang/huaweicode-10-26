import os
from typing import Set, Tuple

import pandas as pd

from Classifiers.TrainToTest import ModelTrainAndTest
from utils.DataFrameOperation import mergeDataFrames
from utils.DataScripts import getDFmean
from utils.DefineData import TIME_COLUMN_NAME, PID_FEATURE, CPU_FEATURE, FAULT_FLAG
from utils.FileSaveRead import saveDFListToFiles
from utils.auto_forecast import getfilespath, getfilepd, differenceProcess, add_cpu_column, differenceServer, \
    standardLists, changeTimeTo_pdlists, processpdsList, serverpdsList, deal_serverpds_and_processpds, \
    predictAllAbnormal, analysePredictResult, removeAllHeadTail, remove_Abnormal_Head_Tail




"""
将一个DataFrame里面的数值全部更改为对应的模数
"""

def changePDfaultFlag(df: pd.DataFrame) -> pd.DataFrame:
    pdlists = []
    for i, ipd in df.groupby(FAULT_FLAG):
        ipd[FAULT_FLAG] = (i // 10) * 10
        pdlists.append(ipd)
    respd, _ = mergeDataFrames(pdlists)
    return respd



"""
全指标预测
使用低强度的内存带宽异常预测高强度的内存带宽异常
包括训练模型和对模型进行预测, 不进行标准化
"""
if __name__ == "__main__":
    # ============================================================================================= 输入数据定义
    # 先将所有的server文件和process文件进行指定
    # 其中单个server文件我默认是连续的
    predictdirpath = R"C:\Users\lWX1084330\Desktop\正常和异常数据\grapes数据\测试数据-E5-异常数据"
    predictserverfiles = getfilespath(os.path.join(predictdirpath, "server"))
    # 将一些需要保存的临时信息进行保存路径
    spath = "tmp/Grapes-tmp"
    if not os.path.exists(spath):
        os.makedirs(spath)
    # 是否有存在faultFlag
    isExistFaultFlag = True
    # 核心数据 如果isManuallyspecifyCoreList==True那么就专门使用我手工指定的数据，如果==False，那么我使用的数据就是从process文件中推理出来的结果
    coresnumber = 104  # 运行操作系统的实际核心数  如实填写
    isManuallyspecifyCoreList = False  # 默认自动生成
    wrfruncoresnumber = 104  # wrf实际运行在的核心数，如果isManuallyspecifyCoreList = False将会手工推导演
    coresSet = set(range(0, 103))  # wrf实际运行在的核心数

    # 需要对server数据进行处理的指标
    server_feature = [
        # "time",
        "user",
        "nice",
        "system",
        "idle",
        "iowait",
        "irq",
        "softirq",
        "steal",
        "guest",
        "guest_nice",
        "ctx_switches",
        "interrupts",
        "soft_interrupts",
        "syscalls",
        "freq",
        "load1",
        "load5",
        "load15",
        "total",
        "available",
        "percent",
        "used",
        "free",
        "active",
        "inactive",
        "buffers",
        "cached",
        "handlesNum",
        "pgpgin",
        "pgpgout",
        "fault",
        "majflt",
        "pgscank",
        "pgsteal",
        "pgfree",
        # "faultFlag",
    ]
    server_accumulate_feature = ['idle', 'iowait', 'interrupts', 'user', 'system', 'ctx_switches', 'soft_interrupts', 'irq',
                  'softirq', 'steal', 'syscalls', 'handlesNum', 'pgpgin', 'pgpgout', 'fault', 'majflt', 'pgscank',
                  'pgsteal', 'pgfree']

    # 在处理时间格式的时候使用，都被转化为'%Y-%m-%d %H:%M:00' 在这里默认所有的进程数据是同一种时间格式，
    server_time_format = '%Y/%m/%d %H:%M'
    process_time_format = '%Y/%m/%d %H:%M'

    # ============================================================================================= 先将正常数据和预测数据的指标从磁盘中加载到内存中 OK
    print("将数据从文件中读取".center(40, "*"))
    predictprocesspds = []
    predictserverpds = []

    # 加入time faultFlag特征值
    time_server_feature = server_feature.copy()
    # 加入时间是为了process和server的对应， 加入pid 是为了进行分类。加入CPU是为了预测哪个CPU出现了异常
    time_server_feature.extend([TIME_COLUMN_NAME])
    time_server_feature.append(FAULT_FLAG)


    # 预测服务数据
    for ifile in predictserverfiles:
        tpd = getfilepd(ifile)
        tpd = tpd.loc[:, time_server_feature]
        predictserverpds.append(tpd)
    # ============================================================================================= 进行差分OK
    print("对读取到的原始数据进行差分".format(40, "*"))


    # 对异常server服务数据进行差分处理之后，得到一些指标
    predictserverpds = differenceServer(predictserverpds, server_accumulate_feature)

    # ============================================================================================= 求出server里面各个数值的平均值
    allprocesspds, _ = mergeDataFrames(predictserverpds) # 合并
    # 将所有的错误都掐头去尾
    allfeatureMeanValue = removeAllHeadTail(allprocesspds)
    # 将所有的错误分离开来
    fault_DataFrameDict = dict(list(allfeatureMeanValue.groupby(FAULT_FLAG)))
    # 得到0
    normalserver_meanvalue = fault_DataFrameDict[0][server_feature].mean()
    normalserver_meanvalue:pd.Series
    normalserver_meanvalue.to_csv(os.path.join(spath, "平均值修改前.csv"))
    # 将used和pgfree修改为指定值
    normalserver_meanvalue["used"] = 56000000000
    normalserver_meanvalue["pgfree"] = 56000000
    normalserver_meanvalue.to_csv(os.path.join(spath, "平均值修改后.csv"))

    # ============================================================================================= 对要预测的数据进行标准化处理
    # 标准化process 和 server数据， 对于process数据，先将cpu想加在一起，然后在求平均值。
    print("标准化要预测的process和server数据".center(40, "*"))
    standard_server_pds = standardLists(pds=predictserverpds, standardFeatures=server_feature,
                                        meanValue=normalserver_meanvalue, standardValue=100)
    # 对标准化结果进行存储
    tpath = os.path.join(spath, "2. 标准化数据存储")
    saveDFListToFiles(os.path.join(tpath, "server_standard"), standard_server_pds)
    # ============================================================================================= 对process数据和server数据进行秒数的处理，将秒数去掉
    standard_server_pds = changeTimeTo_pdlists(standard_server_pds, server_time_format)
    # ============================================================================================= 对process数据和server数据进行特征提取
    print("对server数据进行特征处理".center(40, "*"))
    tpath = os.path.join(spath, "4. server特征提取数据")
    extraction_server_pds = serverpdsList(standard_server_pds, extractFeatures=server_feature,
                                          windowsSize=3, spath=tpath)

    # ============================================================================================= 对process数据和server数据进行特征提取
    # 将训练数据和预测数据摘取下来
    print("将训练和预测数据分离开".center(40, "*"))
    allprocesspds, _ = mergeDataFrames(extraction_server_pds) # 合并数据
    # ========去除54 55，得到训练数据
    allTrainedPD = remove_Abnormal_Head_Tail(allprocesspds, abnormals={54, 55}) # 先去掉54和55及其首尾
    allTrainedPD = removeAllHeadTail(allTrainedPD) # 去掉剩下的首尾部分
    # ========只得到54 55 当作测试数据
    allTestPD = removeAllHeadTail(allprocesspds)
    fault_DataFrameDict = dict(list(allTestPD.groupby(FAULT_FLAG)))
    allTestPD, _ = mergeDataFrames([fault_DataFrameDict[54], fault_DataFrameDict[55]])

    # 将标签纸更改一下
    allTrainedPD = changePDfaultFlag(allTrainedPD)
    allTestPD = changePDfaultFlag(allTestPD)

    allTrainedPD.to_csv(os.path.join(spath, "训练数据.csv"))
    allTestPD.to_csv(os.path.join(spath, "测试数据.csv"))

    # ============================================================================================= 模型的训练和预测
    allfeatureload1_nosuffix = list(allTestPD.columns)
    max_depth = 5
    ModelTrainAndTest(allTrainedPD, allTestPD, spath=spath, selectedFeature=allfeatureload1_nosuffix, modelpath="tmp/grapemodels/memory_bandwidth_model", maxdepth=max_depth)







































































































