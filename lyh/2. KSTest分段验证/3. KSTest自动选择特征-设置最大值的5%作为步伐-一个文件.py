import os
from typing import List, Dict

import numpy as np
import pandas as pd

from utils.DataFrameOperation import mergeDataFrames
from utils.DefineData import FAULT_FLAG, TIME_COLUMN_NAME
from utils.FeatureSelection import getPValueFromTwoDF, getFeatureNameByBenjamini_Yekutiel
from utils.FileSaveRead import saveDFListToFiles
from utils.auto_forecast import getfilespath, getfilepd, differenceServer, removeAllHeadTail, smooth_dfs

"""
得到一个异常前后的正常值
# 确定
"""
def getNormalPD(abnormalserverPD: pd.DataFrame, abnormaltype: int):
    flags = list(abnormalserverPD[FAULT_FLAG])
    selectflag = [False] * len(flags)
    faultbeginpos = -1
    faultendpos = -1
    for i, fault in enumerate(flags):
        if fault == abnormaltype:
            faultendpos = i
        if fault == abnormaltype and faultbeginpos == -1:
            faultbeginpos = i
    # 将前面的正常标签找到
    normalpos = faultbeginpos - 1
    while normalpos >= 0:
        if flags[normalpos] != 0:
            break
        selectflag[normalpos] = True
        normalpos -= 1

    normalpos = faultendpos + 1
    while normalpos < len(flags):
        if flags[normalpos] != 0:
            break
        selectflag[normalpos] = True
        normalpos += 1
    return abnormalserverPD[selectflag]

"""
比较两个server的DataFrame 异常为abnormal
"""

def getServerDiffFeaturesFromOneData(abnormalserverPD: pd.DataFrame, abnormaltype: int, by_fdr: float = 0.05) -> Dict[str, float]:
    abnormalserverPD = removeAllHeadTail(abnormalserverPD, windowsize=3)
    abnormal_pdDict = dict(list(abnormalserverPD.groupby(FAULT_FLAG))) # 指定异常类型
    specialAbnormalPd  = abnormal_pdDict[abnormaltype] # 异常类型
    # normalserverPD = abnormal_pdDict[0]
    normalserverPD = getNormalPD(abnormalserverPD=abnormalserverPD, abnormaltype=abnormaltype)
    print("正常数据长度：{}".format(len(normalserverPD)))
    print("异常特征{} 长度：{}".format(abnormaltype, len(specialAbnormalPd)))

    # =====================================================得到特征选择
    feature_pvalueDict = getPValueFromTwoDF(normalserverPD, specialAbnormalPd)
    selectFeatureDict, notselectFeatureDict = getFeatureNameByBenjamini_Yekutiel(feature_pvalueDict, fdr=by_fdr)
    selectFeatureList = sorted(selectFeatureDict.items(), key=lambda item: item[1], reverse=False)
    notselectFeatureList = sorted(notselectFeatureDict.items(), key=lambda item: item[1], reverse=False)
    print("特征总个数:{}".format(len(selectFeatureList) + len(notselectFeatureList)))
    print("选择的特征".center(40, "*"))
    print("个数: {}".format(len(selectFeatureList)))
    print(selectFeatureList)
    print("没有选择的特征".center(40, "*"))
    print("个数：{}".format(len(notselectFeatureList)))
    print(notselectFeatureList)
    return selectFeatureDict


"""
将一个DataFrame中的各个元素进行分割
保证feaMaxValue的key元素在里面
会修改元素本身
"""


def cutOneDataframe(onepd: pd.DataFrame, serverfeature: List[str], feaMaxValue: pd.Series, percentstep: float):
    def oneColumnsFunction(x: pd.Series):
        # 得到最大值
        columnname = x.name
        maxvalues = feaMaxValue[columnname]
        steps = maxvalues * percentstep
        bins = list(np.arange(0, maxvalues, steps))
        bins.append((maxvalues))
        labels = bins[0: -1]
        return pd.cut(x, bins=bins, labels=labels, right=False)
    onepd[serverfeature] = onepd[serverfeature].apply(oneColumnsFunction)


"""
将一个DataFrame列表按照比例进行分割
"""


def cutDataframeLists(pds: List[pd.DataFrame], serverfeature: List[str], feaMaxValue: pd.Series, percentstep: float):
    for ipd in pds:
        cutOneDataframe(ipd, serverfeature=serverfeature, feaMaxValue=feaMaxValue, percentstep=percentstep)


if __name__ == "__main__":
    predictdirpath = R"C:\Users\lWX1084330\Desktop\正常和异常数据\训练数据-E5-1km-异常数据"
    predictserverfiles = getfilespath(os.path.join(predictdirpath, "server"))
    predictprocessfiles = getfilespath(os.path.join(predictdirpath, "process"))

    abnormalType = 55  # 要比较的异常类型
    spath = "tmp/特征比较"  # 结果文件输出的路径
    # 为wrf运行时使用的核心数量
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
    server_accumulate_feature = ['idle', 'iowait', 'interrupts', 'user', 'system', 'ctx_switches', 'soft_interrupts',
                                 'irq',
                                 'softirq', 'steal', 'syscalls', 'handlesNum', 'pgpgin', 'pgpgout', 'fault', 'majflt',
                                 'pgscank',
                                 'pgsteal', 'pgfree']
    # 每个特征值的取值返回，如果在这里，就按照这个范围进行使用，否则不在这个范围内的数值，会自动进行推导求出返回。
    # 格式为featurename: (lengthstep) lengthstep代表着多少长度以内视为同一个数字
    defaultPercentstep = 0.1  # 默认使用最大的值的0.05作为一个步
    BY_FDR = 0.05

    # ============================================================================================= 先将正常数据和预测数据的指标从磁盘中加载到内存中
    print("将数据从文件中读取".center(40, "*"))
    predictserverpds = []
    # 添加上时间和FAULT_FLAG
    time_server_feature = server_feature.copy()
    time_server_feature.extend([TIME_COLUMN_NAME, FAULT_FLAG])
    for ifile in predictserverfiles:
        tpd = getfilepd(ifile)
        tpd = tpd.loc[:, time_server_feature]
        predictserverpds.append(tpd)
    # ============================================================================================= 对server文件中的累积指标进行差分
    print("将server数据中的累积指标进行差分处理".center(40, "*"))
    difference_abnormal_serverpds = differenceServer(serverpds=predictserverpds,
                                                     accumulateFeatures=server_accumulate_feature)
    # ============================================================================================= 对server文件中进行平滑处理
    print("将server数据中的指标进行平滑处理".center(40, "*"))
    smooth_dfs(difference_abnormal_serverpds)
    # ============================================================================================= 对server文件中最大值和最小值的
    allserverpd, _ = mergeDataFrames(difference_abnormal_serverpds)
    serverminValue = allserverpd[server_feature].min()
    servermaxValue = allserverpd[server_feature].max()

    # 将max的值都增大5%
    servermaxValue = servermaxValue * 1.05
    servermaxValue = servermaxValue + 1

    # ============================================================================================= 将每一列的最大值的5% 或者 其他数值作为一步
    cutDataframeLists(difference_abnormal_serverpds, serverfeature=server_feature, feaMaxValue=servermaxValue,
                      percentstep=defaultPercentstep)
    saveDFListToFiles(os.path.join(spath, "异常server"), difference_abnormal_serverpds)
    # ============================================================================================= KSTest进行比较
    allabnormalpds, _ = mergeDataFrames(difference_abnormal_serverpds)
    getServerDiffFeaturesFromOneData(abnormalserverPD=allabnormalpds, abnormaltype=abnormalType, by_fdr=BY_FDR)










