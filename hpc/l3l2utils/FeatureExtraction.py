import os
from typing import List, Union

import numpy as np
import pandas as pd

from hpc.l3l2utils.DataFrameOperation import mergeDataFrames
from hpc.l3l2utils.DataOperation import pushLabelToEnd, pushLabelToFirst, sortLabels
from hpc.l3l2utils.DefineData import PID_FEATURE, TIME_COLUMN_NAME, FAULT_FLAG

"""
-   功能介绍：
    将一个DataFrame中的所有行中指定的特征值都都剪去第一行，
    其中第一行的数据等于0
-   返回值一个DataFrame

-   重点：传入的参数必须是index=[0, 1, 2]
    传入前可以通过reset_index(drop=True, inplace=True)
"""


def subtractLastLineFromDataFrame(df: pd.DataFrame, columns: List) -> Union[None, pd.DataFrame]:
    if len(columns) == 0:
        return df
    df = df.copy()
    if len(df) <= 1:
        return df
    df.reset_index(drop=True, inplace=True)
    # 先将整个表格往上一隔
    dfcolumns_1 = df.loc[:, columns].shift(periods=-1, axis=0, fill_value=0)
    # 然后相减
    dfcolumns_2 = dfcolumns_1 - df.loc[:, columns]
    # 然后下一一位
    df.loc[:, columns] = dfcolumns_2.shift(periods=1, axis=0, fill_value=0)
    # 然后需要让第一行和第二行进行相等
    df.loc[0, columns] = df.loc[1, columns]
    return df


"""
对进程数据进行差分, 其他数据全部保存不变
进程数据的差分是按照pid进行修改的
"""


def differenceProcess(processpds: List[pd.DataFrame], accumulateFeatures: List[str]) -> List[pd.DataFrame]:
    if len(accumulateFeatures) == 0:
        return processpds
    differencepds = []
    for iprocesspd in processpds:
        subtractpdLists = []
        for ipid, ipd in iprocesspd.groupby(PID_FEATURE):
            # 先将一些不可用的数据进行清除,比如一个进程只运行了两分钟
            subtractpd = subtractLastLineFromDataFrame(ipd, columns=accumulateFeatures)
            subtractpdLists.append(subtractpd)
        allsubtractpd = mergeDataFrames(subtractpdLists)
        differencepds.append(allsubtractpd)
    return differencepds


"""
对数据进行差分处理
并且对pgfree这个指标进行中位数平滑
"""


def differenceServer(serverpds: List[pd.DataFrame], accumulateFeatures: List[str]) -> List[pd.DataFrame]:
    """
    对server数据列表中pgfree进行滑动窗口的处理
    会将传入参数的列表中dataframe本身数值进行修改
    """


    # ============函数运行
    if len(accumulateFeatures) == 0:
        return serverpds
    differencepds = []
    for iserverpd in serverpds:
        subtractpd = subtractLastLineFromDataFrame(iserverpd, columns=accumulateFeatures)
        differencepds.append(subtractpd)
    # 中位数进行平滑操作
    # differencepds = smooth_pgfree(differencepds, smoothwinsize=7)
    return differencepds


"""
将一个DataFrame的值进行标准化
如果meanValue为空，那就取所有数据的平均值
"""


def standardDataFrame(df: pd.DataFrame, standardFeatures=None, meanValue=None,
                      standardValue: int = 100, standardValueType: str = "int64") -> pd.DataFrame:
    if standardFeatures is None:
        standardFeatures = []
    # 如果为空 代表使用自己的mean
    if meanValue is None:
        meanValue = df.mean()
    # 需要先将meanValue中的0值去掉
    meanValue = meanValue[~meanValue.isin([0])]
    columnnames = list(set(meanValue.index) & set(standardFeatures))
    df.loc[:, columnnames] = (df.loc[:, columnnames] / meanValue[columnnames] * standardValue).astype(standardValueType)
    return df


"""
将一个列表中的DataFrame的数值根据平均值进行标准化 
"""


def standardLists(pds: List[pd.DataFrame], standardFeatures: List[str], meanValue, standardValue: int = 100) -> List[
    pd.DataFrame]:
    if len(standardFeatures) == 0: # 如果没有需要提取的特征值，那就直接返回
        return pds
    # ==== 函数开始运行
    standardList = []
    for ipd in pds:
        tpd = standardDataFrame(ipd, standardFeatures, meanValue, standardValue)
        standardList.append(tpd)
    return standardList


"""

"""


def featureExtractionPd(df: pd.DataFrame, extraFeature: List[str], windowSize: int = 5) -> pd.DataFrame:
    # 一个用于判断x的百分比的值
    def quantile(n):
        def quantile_(x):
            return np.quantile(x, q=n)

        quantile_.__name__ = 'percentage%d' % (n * 100)
        return quantile_

    # 得到的这个DataFrame是一个二级列名类似下面这种
    #	    system	                            ｜                      user
    #  sum	mean	amax	amin	quantile50  ｜ sum	mean	amax	amin	quantile50
    # 由于max和min的函数名字是amax和amin 所以需要修改，记得还原，防止引起不必要的麻烦
    maxname = np.max.__name__
    minname = np.min.__name__
    np.max.__name__ = "max"
    np.min.__name__ = "min"
    featureExtractionDf = df.loc[:, extraFeature].rolling(window=windowSize, min_periods=1, center=True).agg(
        [np.mean, np.max, np.min, quantile(0.5)])
    np.max.__name__ = maxname
    np.min.__name__ = minname

    # 将二级索引变成一级的
    featureExtractionDf.columns = ["_".join(x) for x in featureExtractionDf.columns]
    # 此时应该将其和原本的df进行合并，这样既保存了原文件，也保存了新的数据
    resdf = pd.concat([df, featureExtractionDf], axis=1)
    resdf = resdf.dropna()
    # 标签进行排序
    resdf = sortLabels(resdf)
    resdf = pushLabelToFirst(resdf, TIME_COLUMN_NAME)
    resdf = pushLabelToEnd(resdf, FAULT_FLAG)
    resdf.dropna(inplace=True)
    resdf.reset_index(drop=True, inplace=True)
    return resdf


"""
函数功能：特征提取一个进程数据Dataframe
如何特征提取的每个pid数据数量都必须删除前三个和后三个
"""


def extractionOneProcessPd(processpd: pd.DataFrame, extractFeatures: List[str],
                           windowsSize: int = 3, spath: str = None) -> pd.DataFrame:
    if spath is not None and not os.path.exists(spath):
        os.makedirs(spath)
    pidpds = []
    print(PID_FEATURE.center(40, "*"))
    for ipid, idf in processpd.groupby(PID_FEATURE):
        print("pid: {} ".format(ipid), end="")
        assert len(idf) > 6  # 对每一个进程开始的前两个点和后两个点都去掉
        idf = idf.iloc[3:-3]  # 删除数据了
        print("size: {}".format(len(idf)))
        # 对对应的特征进行提取
        featureExtractionDf = featureExtractionPd(idf, extraFeature=extractFeatures, windowSize=windowsSize)
        # 将特征提取之后的效果进行保存
        if spath is not None:
            if not os.path.exists(spath):
                os.makedirs(spath)
            featureExtractionDf.to_csv(os.path.join(spath, "{}.csv".format(ipid)))
        pidpds.append(featureExtractionDf)
    allpidpds = mergeDataFrames(pidpds)
    return allpidpds


"""
函数功能：将包含多个process dataframe进行特征提取
"""


def extractionProcessPdLists(processpds: List[pd.DataFrame], extractFeatures: List[str],
                             windowsSize: int = 3, spath: str = None) -> List[pd.DataFrame]:
    if len(extractFeatures) == 0:
        return processpds
    featureExtractiondfs = []
    for ipd in processpds:
        tpd = extractionOneProcessPd(ipd, extractFeatures, windowsSize, spath)
        featureExtractiondfs.append(tpd)
    return featureExtractiondfs


"""
函数功能：将包含多个server dataframe进行特征提取
"""


def extractionServerPdLists(serverpds: List[pd.DataFrame], extractFeatures: List[str],
                            windowsSize: int = 3, spath: str = None) -> List[pd.DataFrame]:
    if len(extractFeatures) == 0:
        return serverpds
    extraction_dfs = []
    for i, iserverpd in enumerate(serverpds):
        # 对累计的特征值进行数据的处理, 默认一个server数据里面都是连续的, 就算不连续，也只会影响几个点
        # subtractpd = subtractLastLineFromDataFrame(iserverpd, columns=accumulateFeatures)
        # 对特征值进行特征提取
        featureExtractionDf = featureExtractionPd(iserverpd, extraFeature=extractFeatures, windowSize=windowsSize)
        if spath is not None:
            if not os.path.exists(spath):
                os.makedirs(spath)
            featureExtractionDf.to_csv(os.path.join(spath, "server" + str(i) + ".csv"))
        extraction_dfs.append(featureExtractionDf)
    return extraction_dfs
