import os
from typing import Dict, List

import pandas as pd

from utils.DataFrameOperation import mergeDataFrames
from utils.DataScripts import getDFmean, mergeTwoDF
from utils.DefineData import TIME_COLUMN_NAME, PID_FEATURE, CPU_FEATURE, FAULT_FLAG
from utils.FileSaveRead import saveDFListToFiles, saveFaultyCoreDict, saveFaultyDict
from utils.auto_forecast import getfilespath, getfilepd, differenceProcess, add_cpu_column, differenceServer, \
    standardLists, changeTimeTo_pdlists, processpdsList, serverpdsList, deal_serverpds_and_processpds, \
    predictAllAbnormal, analysePredictResult, removeAllHeadTail, allMistakesOnExtractingAllCore


"""
将一个含有多个时间段的已经经过差分处理过的process数据传入进去，然后得到这个时间段内的数据总和
sumfeature: 记录要求和的特征值
"""
def getProcessCoreSum(processpd: pd.DataFrame, sumfeature: List[str] = None) -> pd.DataFrame:
    # 根据时间求和
    if sumfeature is None:
        sumfeature = list(processpd.columns)
        if TIME_COLUMN_NAME in sumfeature:
            sumfeature.remove(TIME_COLUMN_NAME)
        if FAULT_FLAG in sumfeature:
            sumfeature.remove(FAULT_FLAG)
    # 将指定指标求和 得到dataframe
    sumfeaturepd = processpd.groupby(TIME_COLUMN_NAME, as_index=False)[sumfeature].sum()
    # 求出指定时间的FaultFlag的
    flagpd = processpd.groupby(TIME_COLUMN_NAME, as_index=False)[FAULT_FLAG].first()
    # 将两者合并在一起
    mergedpd = pd.merge(left=sumfeaturepd, right=flagpd, on=TIME_COLUMN_NAME)
    return mergedpd





# def differenceProcess(processpds: List[pd.DataFrame], accumulateFeatures: List[str]) -> List[pd.DataFrame]:
#     differencepds = []
#     for iprocesspd in processpds:
#         subtractpdLists = []
#         for ipid, ipd in iprocesspd.groupby(PID_FEATURE):
#             # 先将一些不可用的数据进行清除,比如一个进程只运行了两分钟
#             if len(ipd) <= 6:
#                 continue
#             subtractpd = subtractLastLineFromDataFrame(ipd, columns=accumulateFeatures)
#             subtractpdLists.append(subtractpd)
#         allsubtractpd, _ = mergeDataFrames(subtractpdLists)
#         differencepds.append(allsubtractpd)
#     return differencepds



# 主要用于将process的数据每个时刻的cpu数据合并起来
if __name__ == "__main__":
    # ============================================================================================= 输入数据定义
    # 先将所有的server文件和process文件进行指定
    # 其中单个server文件我默认是连续的
    predictdirpath = R"C:\Users\lWX1084330\Desktop\正常和异常数据\训练数据-E5-1km-异常数据"
    predictprocessfiles = getfilespath(os.path.join(predictdirpath, "process"))
    # 指定正常server和process文件路径
    processcpu_modelpath = ""
    # 将一些需要保存的临时信息进行保存路径
    spath = "tmp/processall_data/训练数据-E5-1km-异常数据"
    # 需要对process数据进行处理的指标, cpu数据要在数据部分添加, 在后面，会往这个列表中添加一个cpu数据
    process_feature = ["user", "system", "iowait", "memory_percent", "rss", "vms", "shared",
                       "text", "lib", "data", "dirty", "read_count", "write_count", "read_bytes", "write_bytes",
                       "read_chars", "write_chars", "num_threads"]
    process_accumulate_feature = ["user", "system"]

    # "user",
    # "system",
    # # "children_user",
    # # "children_system",
    # "iowait",
    # # "cpu_affinity",  # 依照这个来为数据进行分类
    # "memory_percent",
    # "rss",
    # "vms",
    # "shared",
    # "text",
    # "lib",
    # "data",
    # "dirty",
    # "read_count",
    # "write_count",
    # "read_bytes",
    # "write_bytes",
    # "read_chars",
    # "write_chars",
    # "num_threads",
    # "voluntary",
    # "involuntary",

    # 在处理时间格式的时候使用，都被转化为'%Y-%m-%d %H:%M:00' 在这里默认所有的进程数据是同一种时间格式，
    process_time_format = '%Y/%m/%d %H:%M'
    # ============================================================================================= 先将正常数据和预测数据的指标从磁盘中加载到内存中
    print("将数据从文件中读取".center(40, "*"))
    predictprocesspds = []
    # 加入time faultFlag特征值
    time_process_feature = process_feature.copy()
    # 加入时间是为了process和server的对应， 加入pid 是为了进行分类。加入CPU是为了预测哪个CPU出现了异常
    time_process_feature.extend([TIME_COLUMN_NAME, PID_FEATURE, CPU_FEATURE])
    # flagFault要视情况而定
    time_process_feature.append(FAULT_FLAG)

    # 预测进程数据 假设同一个进程数据不回分布在两个文件中，每一个文件都是一个完整的文件
    for ifile in predictprocessfiles:
        tpd = getfilepd(ifile)
        tpd = tpd.loc[:, time_process_feature]
        predictprocesspds.append(tpd)
    # ============================================================================================= 对读取到的数据进行差分，并且将cpu添加到要提取的特征中
    print("对读取到的原始数据进行差分".format(40, "*"))
    # 对异常数据进行差分处理之后，得到cpu特征值
    predictprocesspds = differenceProcess(predictprocesspds, process_accumulate_feature)
    # ============================================================================================= 修改时间
    predictprocesspds = changeTimeTo_pdlists(predictprocesspds, process_time_format)
    # # =========================================================================================== 提取每个文件中的和
    # 需要读取特征的累计值
    sumpds = []
    for ipd in predictprocesspds:
        tpd = getProcessCoreSum(ipd, sumfeature=None)
        sumpds.append(tpd)
    # =========================================================================================== 将process数据进行深层次处理
    for ipd in sumpds:
        ipd["cpu"] = ipd["user"] + ipd["system"]

    # =========================================================================================== 将读取到的数据进行保存
    saveDFListToFiles(spath, sumpds)







































































































