
from typing import Dict

import pandas as pd

from hpc.l3l2utils.DataFrameOperation import mergeProceeDF, getSeriesFrequencyMean
from hpc.l3l2utils.DefineData import FAULT_FLAG
from hpc.l3l2utils.ParsingJson import getNormalServerMean



"""
函数功能：
1.存储cpu_change
2.原始pgfree
3.补偿之后的pgfree
4.自动分析得到的pgfree的平均值
5.如果有正常数据，显示正常数据的平均值
"""
def getMemoryBandwidth50Debuginfo(serverpd: pd.DataFrame, processpd: pd.DataFrame, topdownpd: pd.DataFrame) -> pd.DataFrame:
    debugpd = pd.DataFrame()
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
        pgfree_mean = iserverpd["pgfree"].iloc[0:10].mean()["pgfree"]
        # if detectionJson["RequestData"]["type"] == "grapes":
        #     pgfree_mean = iserverpd["pgfree"].iloc[15:17]
        debugpd["pgfree_mean"] = pgfree_mean#debugpd
        debugpd["pgfree_mean_fre"] = getSeriesFrequencyMean(iserverpd["pgfree"])
        iserverpd[cname] = iserverpd[cname] + pgfree_mean * changes
        iserverpd[cname] = iserverpd[cname].rolling(window=5, center=True, min_periods=1).median() # 对pgfree得到的结果重新去掉最大值最小值
        # pgfree 需要减去平均值
        iserverpd[cname] = iserverpd[cname] - pgfree_mean
        return iserverpd
    # ==========================
    compensatePgfree(serverpd, topdownpd, processpd)
    debugpd[FAULT_FLAG] = serverpd[FAULT_FLAG]
    return debugpd


if __name__ == "__main__":
