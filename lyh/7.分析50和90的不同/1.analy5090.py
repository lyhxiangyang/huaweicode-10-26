from typing import List, Dict, Tuple, Any

import pandas as pd

from hpc.l3l2utils.DataFrameOperation import mergeinnerTwoDataFrame, mergeDataFrames
from hpc.l3l2utils.DataFrameSaveRead import getfilepd, savepdfile
from hpc.l3l2utils.DefineData import TIME_COLUMN_NAME
from hpc.l3l2utils.FeatureExtraction import differenceServer
from hpc.l3l2utils.L2L3Main import removeUselessDataFromTopdownList
from hpc.l3l2utils.ParsingJson import getNormalTopdownMean, getNormalServerMean

topdownfilepath = [
    "DATA/2022-01-14新的测试数据/1.wrf_1km_multi_l3/centos11/topdown/topdown.csv", # 1km异常
    "DATA/2022-01-14新的测试数据/3.wrf_3km_multi_l3/centos11/topdown/topdown.csv",  # 3km异常
    "DATA/2022-01-14新的测试数据/4.wrf_9km_multi_L3/centos11/topdown/topdown.csv",  # 9km异常
    "DATA/2022-01-14新的测试数据/22.grapes_test1p_multi_l3/centos11/topdown/topdown.csv",  # grape异常
    "DATA/2022-01-14新的测试数据/2.wrf_1km_multi_normal/centos11/topdown/topdown.csv", # 1km正常
    "DATA/2022-01-14新的测试数据/21.grapes_test1p_multi_normal/centos11/topdown/topdown.csv",
    "DATA/2022-01-14新的测试数据/28.grapes_test_multi_l3_1/centos11/topdown/topdown.csv",
    "DATA/2022-01-14新的测试数据/29.grapes_test_multi_l3_2/centos11/topdown/topdown.csv",
]
serverfilepath = [
    "DATA/2022-01-14新的测试数据/1.wrf_1km_multi_l3/centos11/server/metric_server.csv",  # 1km异常
    "DATA/2022-01-14新的测试数据/3.wrf_3km_multi_l3/centos11/server/metric_server.csv",  # 3km异常
    "DATA/2022-01-14新的测试数据/4.wrf_9km_multi_L3/centos11/server/metric_server.csv",  # 9km异常
    "DATA/2022-01-14新的测试数据/22.grapes_test1p_multi_l3/centos11/server/metric_server.csv",  # grape异常
    "DATA/2022-01-14新的测试数据/2.wrf_1km_multi_normal/centos11/server/metric_server.csv",  # 1km正常
    "DATA/2022-01-14新的测试数据/21.grapes_test1p_multi_normal/centos11/server/metric_server.csv",
    "DATA/2022-01-14新的测试数据/28.grapes_test_multi_l3_1/centos11/server/metric_server.csv",
    "DATA/2022-01-14新的测试数据/29.grapes_test_multi_l3_2/centos11/server/metric_server.csv",
]
savefilepath = "tmp/servertopdown"



"""
传入的是合并的server和topdown数据
"""
def dealOneTopDownPD(itopdowndpd: pd.DataFrame, pgfree_mean)->pd.DataFrame:
    cname = "mflops"
    # itopdowndpd = removeUselessDataFromTopdownList([itopdowndpd])[0]
    cname_median = cname + "_median"
    cname_median_mean = cname_median + "_mean"
    itopdowndpd[cname_median] = itopdowndpd[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    itopdowndpd[cname_median_mean] = itopdowndpd[cname_median].rolling(window=5, center=True, min_periods=1).mean()
    mflops_mean = getNormalTopdownMean(None, [itopdowndpd], [cname_median_mean], datanumber=10)[cname_median_mean]
    print("mflops_mean is : {}".format(mflops_mean))
    itopdowndpd["mflops_mean_100"] = [mflops_mean] * len(itopdowndpd)
    # itopdowndpd["mflops_mean_grapes_110"] = [mflops_mean * 1.1] * len(itopdowndpd)
    # itopdowndpd["mflops_mean_wrf_145"] = [mflops_mean * 1.45] * len(itopdowndpd)
    mflops_change = itopdowndpd[cname_median_mean].apply(lambda x: (mflops_mean - x) / mflops_mean if x < mflops_mean else 0)
    itopdowndpd["mflops_change"] = mflops_change


    cname = "pgfree"
    cname_median = cname + "_median"
    cname_median_mean = cname_median + "_mean"
    itopdowndpd[cname_median] = itopdowndpd[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    itopdowndpd[cname_median_mean] = itopdowndpd[cname_median].rolling(window=5, center=True, min_periods=1).mean()
    if pgfree_mean == -1:
        pgfree_mean = getNormalTopdownMean(None, [itopdowndpd], [cname_median_mean], datanumber=10)[cname_median_mean]
    print("pgfree_mean is : {}".format(pgfree_mean))
    itopdowndpd["pgfree_mean_100"] = [pgfree_mean] * len(itopdowndpd)
    itopdowndpd["pgfree_mean_110"] = [pgfree_mean * 1.1] * len(itopdowndpd)
    itopdowndpd["pgfree_mean_145"] = [pgfree_mean * 1.45] * len(itopdowndpd)
    itopdowndpd[cname_median_mean + "_recover"] = itopdowndpd[cname_median_mean] + pgfree_mean * mflops_change

    cname = "ddrc_rd"
    cname_median = cname + "_median"
    cname_median_mean = cname_median + "_mean"
    itopdowndpd[cname_median] = itopdowndpd[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    itopdowndpd[cname_median_mean] = itopdowndpd[cname_median].rolling(window=5, center=True, min_periods=1).mean()
    ddrc_rd_mean = getNormalTopdownMean(None, [itopdowndpd], [cname_median_mean], datanumber=10)[cname_median_mean]
    print("ddrc_rd_mean is : {}".format(ddrc_rd_mean))
    itopdowndpd["ddrc_rd_mean_100"] = [ddrc_rd_mean] * len(itopdowndpd)
    itopdowndpd["ddrc_rd_mean_110"] = [ddrc_rd_mean * 1.1] * len(itopdowndpd)
    itopdowndpd["ddrc_rd_mean_145"] = [ddrc_rd_mean * 1.45] * len(itopdowndpd)
    itopdowndpd[cname_median_mean + "_recover"] = itopdowndpd[cname_median_mean] + ddrc_rd_mean * mflops_change

    cname = "ddrc_wr"
    cname_median = cname + "_median"
    cname_median_mean = cname_median + "_mean"
    itopdowndpd[cname_median] = itopdowndpd[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    itopdowndpd[cname_median_mean] = itopdowndpd[cname_median].rolling(window=5, center=True, min_periods=1).mean()
    ddrc_wr_mean = getNormalTopdownMean(None, [itopdowndpd], [cname_median_mean], datanumber=10)[cname_median_mean]
    print("ddrc_wr_mean is : {}".format(ddrc_wr_mean))
    itopdowndpd["ddrc_wr_mean_100"] = [ddrc_wr_mean] * len(itopdowndpd)
    itopdowndpd["ddrc_wr_mean_110"] = [ddrc_wr_mean * 1.1] * len(itopdowndpd)
    itopdowndpd["ddrc_wr_mean_145"] = [ddrc_wr_mean * 1.45] * len(itopdowndpd)
    itopdowndpd[cname_median_mean + "_recover"] = itopdowndpd[cname_median_mean] + ddrc_wr_mean * mflops_change

    itopdowndpd["rd_wr_sum_mean"] = ddrc_rd_mean + ddrc_wr_mean
    itopdowndpd["rd_wr_sum"] = itopdowndpd["ddrc_rd_median_mean_recover"] + itopdowndpd["ddrc_wr_median_mean_recover"]
    # 平滑一下
    itopdowndpd["rd_wr_sum"] = itopdowndpd["rd_wr_sum"].rolling(window=3, center=True, min_periods=1).median()


    return itopdowndpd


def processServer(iserverpd: pd):
    nserverpd = differenceServer([iserverpd], ["pgfree"])
    return nserverpd

pgfree_means = [-1, -1, -1, -1, -1, -1, 5000000, 5000000]

if __name__ == "__main__":
    alltopdownpds = []
    for i, ipath in enumerate(topdownfilepath):
        itopdownpd = getfilepd(ipath)
        iserverpd = getfilepd(serverfilepath[i])
        iserverpds = processServer(iserverpd)
        iserverpd = iserverpds[0]
        itpd = mergeinnerTwoDataFrame(lpd=iserverpd, rpd=itopdownpd)
        dealpd = dealOneTopDownPD(itpd,  pgfree_means[i])
        alltopdownpds.append(dealpd)

    for i, ipd in enumerate(alltopdownpds):
        savepdfile(ipd, savefilepath, "topdown{}.csv".format(i))
