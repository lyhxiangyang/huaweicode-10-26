import pandas as pd

from hpc.l3l2utils.DataFrameOperation import mergeinnerTwoDataFrame
from hpc.l3l2utils.DataFrameSaveRead import getfilepd, savepdfile
from hpc.l3l2utils.FeatureExtraction import differenceServer

topdownfilepath = [
    "DATA/2022-01-14新的测试数据/1.wrf_1km_multi_l3/centos11/topdown/topdown.csv", # 1km异常
    "DATA/2022-01-14新的测试数据/3.wrf_3km_multi_l3/centos11/topdown/topdown.csv",  # 3km异常
    "DATA/2022-01-14新的测试数据/4.wrf_9km_multi_L3/centos11/topdown/topdown.csv",  # 9km异常
    "DATA/2022-01-14新的测试数据/22.grapes_test1p_multi_l3/centos11/topdown/topdown.csv",  # grape异常
    "DATA/2022-01-14新的测试数据/2.wrf_1km_multi_normal/centos11/topdown/topdown.csv", # 1km正常
    "DATA/2022-01-14新的测试数据/21.grapes_test1p_multi_normal/centos11/topdown/topdown.csv",
    "DATA/2022-01-14新的测试数据/28.grapes_test_multi_l3_1/centos16/topdown/topdown.csv",

]
serverfilepath = [
    "DATA/2022-01-14新的测试数据/1.wrf_1km_multi_l3/centos11/server/metric_server.csv",  # 1km异常
    "DATA/2022-01-14新的测试数据/3.wrf_3km_multi_l3/centos11/server/metric_server.csv",  # 3km异常
    "DATA/2022-01-14新的测试数据/4.wrf_9km_multi_L3/centos11/server/metric_server.csv",  # 9km异常
    "DATA/2022-01-14新的测试数据/22.grapes_test1p_multi_l3/centos11/server/metric_server.csv",  # grape异常
    "DATA/2022-01-14新的测试数据/2.wrf_1km_multi_normal/centos11/server/metric_server.csv",  # 1km正常
    "DATA/2022-01-14新的测试数据/21.grapes_test1p_multi_normal/centos11/server/metric_server.csv",
    "DATA/2022-01-14新的测试数据/28.grapes_test_multi_l3_1/centos11/server/metric_server.csv",
]
savefilepath = "tmp/servertopdown"


def dealOneTopDownPD(topdownpd: pd.DataFrame) -> pd.DataFrame:
    # 对ddrc_rd进行滑动窗口处理
    cname = "ddrc_rd"
    topdownpd[cname + "_sliding"] = topdownpd[cname].rolling(window=5, center=True, min_periods=1).agg("max").astype(
        "int")
    # 对ddrc_rd进行滑动窗口处理
    cname = "ddrc_wr"
    topdownpd[cname + "_sliding"] = topdownpd[cname].rolling(window=5, center=True, min_periods=1).agg("max").astype(
        "int")
    # 对ddrc_rd进行滑动窗口处理
    cname = "llcm"
    topdownpd[cname + "_sliding"] = topdownpd[cname].rolling(window=5, center=True, min_periods=1).agg("max").astype(
        "int")
    # 对ddrc_rd进行滑动窗口处理
    cname = "mflops"
    topdownpd[cname + "_sliding"] = topdownpd[cname].rolling(window=5, center=True, min_periods=1).agg("max").astype(
        "int")

    # mflops平均值
    cname = "mflops_sliding"
    mflops_mean = topdownpd[cname][0:3].mean()
    print("mflops平均值：{}".format(mflops_mean))
    mflops_change = topdownpd[cname].apply(lambda x : (mflops_mean - x) / mflops_mean if x <= mflops_mean else 0 ) # 如果是-20% 那么对应的值应该增加20%
    topdownpd["mflops_change"] = mflops_change

    # 对ddrc_rd进行分析
    cname = "ddrc_rd_sliding"
    ddrc_rd_mean = topdownpd[cname][0:3].mean() # 得到一个正常值
    print("{}平均值：{}".format(cname, ddrc_rd_mean))
    topdownpd[cname+"_recover"] = topdownpd[cname] + ddrc_rd_mean * mflops_change
    topdownpd[cname+"_recover_sliding"] = topdownpd[cname+"_recover"].rolling(window=5, center=True, min_periods=1).agg("max").astype("int")

    # 对ddrc_wr进行分析
    cname = "ddrc_wr_sliding"
    ddrc_rd_mean = topdownpd[cname][0:3].mean() # 得到一个正常值
    print("{}平均值：{}".format(cname, ddrc_rd_mean))
    topdownpd[cname+"_recover"] = topdownpd[cname] + ddrc_rd_mean * mflops_change
    topdownpd[cname+"_recover_sliding"] = topdownpd[cname+"_recover"].rolling(window=5, center=True, min_periods=1).agg("max").astype("int")

    # 将ddrc_wr和ddrc_rd加在一起
    topdownpd["rd_wr_sum"] = topdownpd["ddrc_rd_sliding_recover_sliding"] + topdownpd["ddrc_wr_sliding_recover_sliding"]

    # 对pgfree进行分析
    cname = "pgfree"
    ddrc_rd_mean = topdownpd[cname][0:3].mean() # 得到一个正常值
    print("{}平均值：{}".format(cname, ddrc_rd_mean))
    topdownpd[cname+"_recover"] = topdownpd[cname] + ddrc_rd_mean * mflops_change
    topdownpd[cname+"_recover_sliding"] = topdownpd[cname+"_recover"].rolling(window=5, center=True, min_periods=1).agg("max").astype("int")


    return topdownpd


def processServer(iserverpd: pd):
    nserverpd = differenceServer([iserverpd], ["pgfree"])
    return nserverpd


if __name__ == "__main__":
    alltopdownpds = []
    for i, ipath in enumerate(topdownfilepath):
        itopdownpd = getfilepd(ipath)

        iserverpd = getfilepd(serverfilepath[i])
        iserverpds = processServer(iserverpd)
        iserverpd = iserverpds[0]

        itpd = mergeinnerTwoDataFrame(lpd=iserverpd, rpd=itopdownpd)
        dealpd = dealOneTopDownPD(itpd)
        alltopdownpds.append(dealpd)

    for i, ipd in enumerate(alltopdownpds):
        savepdfile(ipd, savefilepath, "topdown{}.csv".format(i))
