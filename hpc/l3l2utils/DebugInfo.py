
from typing import Dict

import pandas as pd

from hpc.l3l2utils.DataFrameOperation import mergeProceeDF, getSeriesFrequencyMean, smoothseries
from hpc.l3l2utils.DefineData import FAULT_FLAG
from hpc.l3l2utils.ParsingJson import getNormalServerMean, getNormalTopdownMean

"""
函数功能：
1.存储cpu_change
2.原始pgfree
3.补偿之后的pgfree
4.自动分析得到的pgfree的平均值
5.如果有正常数据，显示正常数据的平均值
"""
def getMemoryBandwidth50Debuginfo(serverpd: pd.DataFrame, processpd: pd.DataFrame, topdownpd: pd.DataFrame, inputDict: Dict = None, detectionJson: Dict = None) -> pd.DataFrame:
    debugpd = pd.DataFrame()

    def getMflopschange(itopdownpd: pd.DataFrame) -> pd.Series:
        cname = "mflops"
        # itopdownpd = removeUselessDataFromTopdownList([itopdownpd])[0]
        itopdownpd[cname] = itopdownpd[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
        itopdownpd[cname] = itopdownpd[cname].rolling(window=5, center=True, min_periods=1).mean()
        #1
        debugpd["mflops"] = itopdownpd[cname]

        # 获得平均值
        mflops_mean = getNormalTopdownMean(detectionJson, [itopdownpd], [cname])[cname]

        mflops_normal_iomax = inputDict["maxflopsinio"]
        itopdownpd[cname] = itopdownpd[cname].apply(lambda x: mflops_mean if x < mflops_normal_iomax else x)
        itopdownpd[cname] = itopdownpd[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
        #1
        debugpd["mflops_delete"] = itopdownpd[cname]
        #2
        debugpd["mflops_mean_10"] = itopdownpd[cname].iloc[0:10].mean()
        debugpd["mflops_mean"] = getSeriesFrequencyMean(itopdownpd[cname])

        mflops_normal_iomax = 15000
        mflops_change = itopdownpd[cname].apply(lambda x: (mflops_mean - x) / mflops_mean if x < mflops_mean  else 0)
        # 将较高的mflpos_change抹为0
        # mflops_change.apply(lambda x: if x > )
        itopdownpd["mflops_change"] = mflops_change
        debugpd["mflops_change"] = mflops_change
        return mflops_change



    def getcpuchange(serverpd: pd.DataFrame, processpd: pd.DataFrame)->pd.Series:
        mergeprocesspd = mergeProceeDF(processpd, sumFeatures=["usr_cpu", "kernel_cpu"])

        if "cpu" not in serverpd.columns.tolist():
            serverpd["cpu"] = serverpd["usr_cpu"] + serverpd["kernel_cpu"]
        if "cpu" not in processpd.columns.tolist():
            processpd["cpu"] = processpd["usr_cpu"] + processpd["kernel_cpu"]

        # iprocesspd cpu的加在一起
        mergeprocesspd["cpu"] = mergeprocesspd["usr_cpu"] + mergeprocesspd["kernel_cpu"]
        serverpd["cpu"] = serverpd["usr_cpu"] + serverpd["kernel_cpu"]
        sub_server_process_cpu = serverpd["cpu"] - mergeprocesspd["cpu"]
        # 如果iserverpd["cpu"]是0， 就当作1
        serverpd["cpu"] = serverpd["cpu"].apply(lambda x: 1 if x == 0 else x)
        cpu_change = sub_server_process_cpu / serverpd["cpu"]
        return cpu_change
    # 保证iserverpds和itodownpds时间与时间相互匹配
    def compensatePgfree(iserverpd: pd.DataFrame, itopdowndpd: pd.DataFrame, iprocesspd: pd.DataFrame, detectionJson: Dict, inplace=True):
        assert len(iserverpd) == len(itopdowndpd)
        if inplace:
            iserverpd = iserverpd.copy()
        # 对iprocess和servercpu中的
        cpu_change = getcpuchange(iserverpd, iprocesspd)
        debugpd["cpu_change"] = cpu_change #debugpd
        changes=cpu_change
        cname = "pgfree"
        iserverpd[cname] = iserverpd[cname].rolling(window=5, center=True, min_periods=1).median() # 先将最大最小值去除
        iserverpd[cname] = iserverpd[cname].rolling(window=5, center=True, min_periods=1).median() # 多去一次
        iserverpd[cname] = iserverpd[cname].rolling(window=5, center=True, min_periods=1).mean()
        debugpd["pgfree_smooth"] = iserverpd["pgfree"] #debugpd
        # 对来自的应用进行判断
        # pgfree_mean = getNormalServerMean(detectionJson, [iserverpd], [cname], datanumber=10)[cname]
        # pgfree_mean = getNormalServerMean(detectionJson, [iserverpd], [cname], datanumber=10)[cname]
        pgfree_mean = getSeriesFrequencyMean(iserverpd["pgfree"])
        # if detectionJson["RequestData"]["type"] == "grapes":
        #     pgfree_mean = iserverpd["pgfree"].iloc[15:17]
        debugpd["pgfree_mean"] = getNormalServerMean(detectionJson, [iserverpd], [cname], datanumber=10)[cname]#debugpd
        debugpd["pgfree_mean_10"] = iserverpd[cname].iloc[0:10].mean()

        iserverpd[cname] = iserverpd[cname] + pgfree_mean * changes

        mflopschanges=getMflopschange(itopdowndpd)
        debugpd["pgfree_change_cpu"] = iserverpd[cname]
        debugpd["pgfree_change_mflops"] = iserverpd[cname] + pgfree_mean * mflopschanges

        iserverpd[cname] = iserverpd[cname].rolling(window=5, center=True, min_periods=1).median() # 对pgfree得到的结果重新去掉最大值最小值
        # pgfree 需要减去平均值
        iserverpd[cname] = iserverpd[cname] - pgfree_mean
        return iserverpd
    # ==========================
    compensatePgfree(serverpd, topdownpd, processpd, detectionJson)
    debugpd["time"]=serverpd["time"]
    if inputDict["isExistFaultFlag"]:
        debugpd[FAULT_FLAG] = serverpd[FAULT_FLAG]
    return debugpd




"""
函数功能：
1.存储cpu_change
2.原始pgfree
3.补偿之后的pgfree
4.自动分析得到的pgfree的平均值
5.如果有正常数据，显示正常数据的平均值
"""
def getCache90Debuginfo(serverpd: pd.DataFrame, processpd: pd.DataFrame, topdownpd: pd.DataFrame, inputDict: Dict = None, detectionJson: Dict = None) -> pd.DataFrame:
    debugpd = pd.DataFrame()

    def getMflopschange(itopdownpd: pd.DataFrame) -> pd.Series:
        cname = "mflops"
        # itopdownpd = removeUselessDataFromTopdownList([itopdownpd])[0]
        itopdownpd[cname] = itopdownpd[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
        itopdownpd[cname] = itopdownpd[cname].rolling(window=5, center=True, min_periods=1).mean()
        #1
        debugpd["mflops"] = itopdownpd[cname]

        # 获得平均值
        mflops_mean = getNormalTopdownMean(detectionJson, [itopdownpd], [cname])[cname]

        mflops_normal_iomax = inputDict["maxflopsinio"]
        itopdownpd[cname] = itopdownpd[cname].apply(lambda x: mflops_mean if x < mflops_normal_iomax else x)
        itopdownpd[cname] = itopdownpd[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
        #1
        debugpd["mflops_delete"] = itopdownpd[cname]
        #2
        debugpd["mflops_mean_10"] = itopdownpd[cname].iloc[0:10].mean()
        debugpd["mflops_mean"] = getSeriesFrequencyMean(itopdownpd[cname])

        mflops_normal_iomax = 15000
        mflops_change = itopdownpd[cname].apply(lambda x: (mflops_mean - x) / mflops_mean if x < mflops_mean  else 0)
        # 将较高的mflpos_change抹为0
        # mflops_change.apply(lambda x: if x > )
        itopdownpd["mflops_change"] = mflops_change
        debugpd["mflops_change"] = mflops_change
        return mflops_change



    def getcpuchange(serverpd: pd.DataFrame, processpd: pd.DataFrame)->pd.Series:
        mergeprocesspd = mergeProceeDF(processpd, sumFeatures=["usr_cpu", "kernel_cpu"])

        if "cpu" not in serverpd.columns.tolist():
            serverpd["cpu"] = serverpd["usr_cpu"] + serverpd["kernel_cpu"]
        if "cpu" not in processpd.columns.tolist():
            processpd["cpu"] = processpd["usr_cpu"] + processpd["kernel_cpu"]

        # iprocesspd cpu的加在一起
        mergeprocesspd["cpu"] = mergeprocesspd["usr_cpu"] + mergeprocesspd["kernel_cpu"]
        serverpd["cpu"] = serverpd["usr_cpu"] + serverpd["kernel_cpu"]
        sub_server_process_cpu = serverpd["cpu"] - mergeprocesspd["cpu"]
        # 如果iserverpd["cpu"]是0， 就当作1
        serverpd["cpu"] = serverpd["cpu"].apply(lambda x: 1 if x == 0 else x)
        cpu_change = sub_server_process_cpu / serverpd["cpu"]
        return cpu_change
    # 保证iserverpds和itodownpds时间与时间相互匹配
    def compensateRW(itopdownpd: pd.DataFrame, iserverpd: pd.DataFrame, iprocesspd: pd.DataFrame, detectJson: Dict, inplace=True):
        if inplace:
            itopdownpd = itopdownpd.copy()
        # 对itopdownpd中的mflops进行平滑处理
        mflops_change = getMflopschange(itopdownpd)
        # cpuchange = getcpuchange(iserverpd, iprocesspd)
        change = smoothseries(mflops_change)
        # 对ddrc_rd进行滑动窗口处理
        rd_cname = "ddrc_rd"
        itopdownpd[rd_cname] = itopdownpd[rd_cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
        itopdownpd[rd_cname] = itopdownpd[rd_cname].rolling(window=5, center=True, min_periods=1).mean()
        debugpd["ddrc_rd"] = itopdownpd[rd_cname]
        ddrc_rd_mean = getNormalTopdownMean(detectJson, [itopdownpd], [rd_cname], datanumber=10)[rd_cname]
        itopdownpd[rd_cname] = itopdownpd[rd_cname] + ddrc_rd_mean * change
        debugpd["ddrc_rd_means"] = ddrc_rd_mean
        debugpd["ddrc_rd_mflops"] = itopdownpd[rd_cname]

        # 对ddrc_rd进行滑动窗口处理
        wr_cname = "ddrc_wr"
        itopdownpd[wr_cname] = itopdownpd[wr_cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
        itopdownpd[wr_cname] = itopdownpd[wr_cname].rolling(window=5, center=True, min_periods=1).mean()
        debugpd[wr_cname] = itopdownpd[wr_cname]
        ddrc_rd_mean = getNormalTopdownMean(detectJson, [itopdownpd], [wr_cname], datanumber=10)[wr_cname]
        itopdownpd[wr_cname] = itopdownpd[wr_cname] + ddrc_rd_mean * change
        debugpd["ddrc_wr_means"] = ddrc_rd_mean
        debugpd["ddrc_wr_mflops"] = itopdownpd[wr_cname]

        debugpd["ddrc_ddwr_sum_mean"] = debugpd["ddrc_rd_means"] + debugpd["ddrc_wr_means"]

        # 对rd_wr_sum进行结合 减去平均值  阈值与6000比较
        rd_wr_cname = "ddrc_ddwr_sum"
        itopdownpd[rd_wr_cname] = itopdownpd[rd_cname] + itopdownpd[wr_cname]
        itopdownpd[rd_wr_cname] = itopdownpd[rd_wr_cname].rolling(window=5, center=True, min_periods=1).median()
        debugpd[rd_wr_cname] = itopdownpd[rd_wr_cname]

        rd_wr_sum_mean = getNormalTopdownMean(detectJson, [itopdownpd], [rd_wr_cname], datanumber=10)[rd_wr_cname]
        itopdownpd[rd_wr_cname] = itopdownpd[rd_wr_cname] - rd_wr_sum_mean
        # 重点是mflops、ddrc_rd、ddrc_ddwr_sum
        return itopdownpd
    # ==========================
    compensateRW(serverpd, topdownpd, processpd, detectionJson)
    debugpd["time"]=serverpd["time"]
    if inputDict["isExistFaultFlag"]:
        debugpd[FAULT_FLAG] = serverpd[FAULT_FLAG]
    return debugpd
