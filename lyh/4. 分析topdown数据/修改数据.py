import os
import sys

import pandas as pd

from l3l2utils.DataFrameSaveRead import getfilepd, savepdfile

topdownfilepath = [
    "DATA/2022-01-14新的测试数据/1.wrf_1km_multi_l3/centos11-flag/topdown/topdown.csv"
]
savefilepath = "tmp/topdown"


def dealOneTopDownPD(topdownpd: pd.DataFrame) -> pd.DataFrame:
    # 对ddrc_rd进行滑动窗口处理
    cname = "ddrc_rd"
    topdownpd[cname + "_rolling"] = topdownpd[cname].rolling(window=5, center=True, min_periods=1).agg("max").astype(
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
    mflops_change = topdownpd[cname].apply(lambda x : (mflops_mean - x) / mflops_mean if x <= mflops_mean else 1 ) # 如果是-20% 那么对应的值应该增加20%

    # 对ddrc_rd进行分析
    cname = "ddrc_rd_sliding"
    ddrc_rd_mean = topdownpd[cname][0:3].mean() # 得到一个正常值
    print("{}平均值：{}".format(cname, ddrc_rd_mean))
    topdownpd[cname+"_recover"] = topdownpd[cname] + ddrc_rd_mean * mflops_change


    return topdownpd


if __name__ == "__main__":
    alltopdownpds = []
    for ipath in topdownfilepath:
        itpd = getfilepd(ipath)
        dealpd = dealOneTopDownPD(itpd)
        alltopdownpds.append(dealpd)

    for i, ipd in enumerate(alltopdownpds):
        savepdfile(ipd, savefilepath, "topdown{}.csv".format(i))
