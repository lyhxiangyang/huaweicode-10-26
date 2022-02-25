import pandas as pd

from hpc.l3l2utils.DataFrameOperation import mergeinnerTwoDataFrame
from hpc.l3l2utils.DataFrameSaveRead import getfilepd, savepdfile
from hpc.l3l2utils.FeatureExtraction import differenceServer
from hpc.l3l2utils.L2L3Main import removeUselessDataFromTopdownList

topdownfilepath = [
    "DATA/2022-01-14新的测试数据/28.grapes_test_multi_l3_1/centos11/topdown/topdown.csv",
    "DATA/2022-01-14新的测试数据/29.grapes_test_multi_l3_2/centos11/topdown/topdown.csv",
]
serverfilepath = [
    "DATA/2022-01-14新的测试数据/28.grapes_test_multi_l3_1/centos11/server/metric_server.csv",
    "DATA/2022-01-14新的测试数据/29.grapes_test_multi_l3_2/centos11/server/metric_server.csv",
]
mflops_mean = [38000, 387500]
ddrc_rd_mean = [57300, 57300]
ddrc_wr_mean = [19100, 19100]
pgfree_mean = [4750000, 4750000]
savefilepath = "tmp/servergrapestopdown"


def dealOneTopDownPD(topdownpd: pd.DataFrame, mflops_mean: int, ddrc_rd_mean: int, ddrc_wr_mean: int, pgfree_mean: int) -> pd.DataFrame:
    cname = "mflops"
    cnamemedian = cname + "_median"
    cnamemedianMean = cnamemedian + "_mean"
    topdownpd[cnamemedian] = topdownpd[cname].rolling(window=5, center=True, min_periods=1).median()
    topdownpd[cnamemedianMean] = topdownpd[cnamemedian].rolling(window=5, center=True, min_periods=1).mean()
    # mflops change
    mflops_change = topdownpd[cnamemedianMean].apply(lambda x : (mflops_mean - x) / mflops_mean if x <= mflops_mean else 0 )
    topdownpd["mflops_change"] = mflops_change

    # 对ddrc_rd进行滑动窗口处理
    cname = "ddrc_rd"
    cnamemedian = cname + "_median"
    cnamemedianMean = cnamemedian + "_mean"
    topdownpd[cnamemedian] = topdownpd[cname].rolling(window=5, center=True, min_periods=1).median()
    topdownpd[cnamemedianMean] = topdownpd[cnamemedian].rolling(window=5, center=True, min_periods=1).mean()
    topdownpd[cnamemedianMean + "_recover"] = topdownpd[cnamemedianMean] + ddrc_rd_mean * mflops_change

    cname = "ddrc_wr"
    cnamemedian = cname + "_median"
    cnamemedianMean = cnamemedian + "_mean"
    topdownpd[cnamemedian] = topdownpd[cname].rolling(window=5, center=True, min_periods=1).median()
    topdownpd[cnamemedianMean] = topdownpd[cnamemedian].rolling(window=5, center=True, min_periods=1).mean()
    topdownpd[cnamemedianMean + "_recover"] = topdownpd[cnamemedianMean] + ddrc_wr_mean * mflops_change

    # 将ddrc_wr和ddrc_rd加在一起
    topdownpd["rd_wr_sum"] = topdownpd["ddrc_rd_median_mean_recover"] + topdownpd["ddrc_wr_median_mean_recover"]

    cname = "pgfree"
    cnamemedian = cname + "_median"
    cnamemedianMean = cnamemedian + "_mean"
    topdownpd[cnamemedian] = topdownpd[cname].rolling(window=5, center=True, min_periods=1).median()
    topdownpd[cnamemedianMean] = topdownpd[cnamemedian].rolling(window=5, center=True, min_periods=1).mean()
    topdownpd[cnamemedianMean + "_recover"] = topdownpd[cnamemedianMean] + pgfree_mean * mflops_change

    return topdownpd


def processServer(iserverpd: pd):
    nserverpd = differenceServer([iserverpd], ["pgfree"])
    return nserverpd

# 查看pgfree的变化
if __name__ == "__main__":
    alltopdownpds = []
    for i, ipath in enumerate(topdownfilepath):
        itopdownpd = getfilepd(ipath)

        iserverpd = getfilepd(serverfilepath[i])
        iserverpd = processServer(iserverpd)[0]
        iserverpd = removeUselessDataFromTopdownList([iserverpd])[0]

        itpd = mergeinnerTwoDataFrame(lpd=iserverpd, rpd=itopdownpd)
        dealpd = dealOneTopDownPD(itpd, mflops_mean[i], ddrc_rd_mean[i], ddrc_wr_mean[i], pgfree_mean[i])
        alltopdownpds.append(dealpd)

    for i, ipd in enumerate(alltopdownpds):
        savepdfile(ipd, savefilepath, "topdown{}.csv".format(i))
