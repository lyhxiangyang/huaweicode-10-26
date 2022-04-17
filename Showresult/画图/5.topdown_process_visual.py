import os
import time
from typing import List

import pandas as pd
import plotly.graph_objs as go

from Showresult.显示process数据.processmemory import FAULTFLAG
from hpc.l3l2utils.DataFrameOperation import mergeinnerTwoDataFrame, smoothseries
from hpc.l3l2utils.DataOperation import changeTimeToFromPdlists, getRunHPCTimepdsFromProcess, getsametimepdList
from hpc.l3l2utils.DefineData import TIME_COLUMN_NAME
from hpc.l3l2utils.FeatureExtraction import differenceServer, differenceProcess


def n_cols_plot(df, yy, title):
    data = []
    for y in yy:
        trace1 = go.Scatter(x=df.index, y=df[y], mode='lines', name=y)
        data.append(trace1)
        # abnormal_point = df[(df["faultFlag"] != 0)]
        abnormal_point = df[df["faultFlag"].apply(lambda x: True if x != 0 and x != -1 else False)]
        trace = go.Scatter(x=abnormal_point.index, y=abnormal_point[y], mode='markers')
        data.append(trace)
    layout = dict(title=title)
    fig = go.Figure(data=data, layout=layout)
    fig.show()



def processing(filepath: str, filename: str = None):
    file = filepath
    if filename is not None:
        file = os.path.join(filepath, filename)
    itopdownpd = pd.read_csv(file, index_col=TIME_COLUMN_NAME)
    if FAULTFLAG not in itopdownpd.columns:
        itopdownpd[FAULTFLAG] = 0
    itopdownpd = itopdownpd.dropna()
    # 修改列名 去掉每个文件中的空格
    # mflops
    itopdownpd["mflops_median"] = itopdownpd["mflops"].rolling(window=5, center=True, min_periods=1).median()
    itopdownpd["mflops_median_mean"] = itopdownpd["mflops"].rolling(window=5, center=True, min_periods=1).mean()

    # ddrc_rd
    rd_cname = "ddrc_rd"
    itopdownpd[rd_cname + "_median"] = itopdownpd[rd_cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    itopdownpd[rd_cname + "_mean"] = itopdownpd[rd_cname].rolling(window=5, center=True, min_periods=1).mean()

    wr_cname = "ddrc_wr"
    itopdownpd[wr_cname + "_median"] = itopdownpd[wr_cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    itopdownpd[wr_cname + "_mean"] = itopdownpd[wr_cname].rolling(window=5, center=True, min_periods=1).mean()

    rd_wr_cname = "ddrc_ddwr_sum"
    itopdownpd[rd_wr_cname] = itopdownpd[rd_cname] + itopdownpd[wr_cname]
    itopdownpd[rd_wr_cname + "_median"] = itopdownpd[rd_wr_cname].rolling(window=5, center=True, min_periods=1).median()


    itopdownpd['flag'] = itopdownpd['faultFlag'].apply(lambda x: x % 10)
    itopdownpd = itopdownpd.dropna()
    return itopdownpd

# 得到server和process的pd
def getServerTopdownandProcesspds(filepath: str):
    itopdownpath = os.path.join(filepath,"topdown" , "topdown.csv")
    iprocesspath = os.path.join(filepath, "process","hpc_process.csv")
    iserverpath = os.path.join(filepath, "server", "metric_server.csv")

    # 读取到dataframe中
    itopdownpd = pd.read_csv(itopdownpath)
    iprocesspd = pd.read_csv(iprocesspath)
    iserverpd = pd.read_csv(iserverpath)

    # 对iserver进行时间处理
    serverlists = changeTimeToFromPdlists([iserverpd], isremoveDuplicate=True)
    topdownlists = changeTimeToFromPdlists([itopdownpd], isremoveDuplicate=True)
    processpdlists = changeTimeToFromPdlists([iprocesspd])
    # 对数据进行差分处理
    serverlists = differenceServer(serverlists, ["pgfree", "usr_cpu", "kernel_cpu"])
    # topdownlists = differenceServer(topdownlists, ["pgfree", "usr_cpu", "kernel_cpu"])
    processpdlists = differenceProcess(processpdlists, ["usr_cpu", "kernel_cpu"])
    return serverlists[0], topdownlists[0], processpdlists[0]

def mergeProceeDF(processpd: pd.DataFrame, sumFeatures=None, inplace=True):
    if sumFeatures is None:
        return pd.DataFrame()
    if inplace:
        processpd = processpd.copy()
    if TIME_COLUMN_NAME not in sumFeatures:
        sumFeatures.append(TIME_COLUMN_NAME)
    tpd = processpd[sumFeatures].groupby("time").sum()
    tpd1 = processpd[[TIME_COLUMN_NAME, "pid"]].groupby("time").min()
    tpd.reset_index(drop=False, inplace=True)
    tpd1.reset_index(drop=False, inplace=True)
    respd = mergeinnerTwoDataFrame(lpd=tpd, rpd=tpd1)
    return respd


# 传入的三个数值应该是一样的
def getSUMWR(itopdownpd: pd.DataFrame, iprocesspd: pd.DataFrame, iserverpd: pd.DataFrame):
    mergeprocesspd = mergeProceeDF(iprocesspd, sumFeatures=["rss", "usr_cpu", "kernel_cpu"])

    cname = "mflops"
    itopdownpd[cname] = itopdownpd[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    itopdownpd[cname] = itopdownpd[cname].rolling(window=5, center=True, min_periods=1).mean()
    mflops_mean = itopdownpd[cname][0:10].mean()
    mflops_change = itopdownpd[cname].apply(lambda x: (mflops_mean - x) / mflops_mean if x < mflops_mean and x > 18000 else 0)

    # ddrc_rd
    rd_cname = "ddrc_rd"
    itopdownpd[rd_cname + "_median"] = itopdownpd[rd_cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    itopdownpd[rd_cname + "_mean"] = itopdownpd[rd_cname].rolling(window=5, center=True, min_periods=1).mean()

    wr_cname = "ddrc_wr"
    itopdownpd[wr_cname + "_median"] = itopdownpd[wr_cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    itopdownpd[wr_cname + "_mean"] = itopdownpd[wr_cname].rolling(window=5, center=True, min_periods=1).mean()

    rd_wr_cname = "ddrc_ddwr_sum"
    itopdownpd[rd_wr_cname] = itopdownpd[rd_cname] + itopdownpd[wr_cname]
    itopdownpd[rd_wr_cname + "_median"] = itopdownpd[rd_wr_cname].rolling(window=5, center=True, min_periods=1).median()


    # iprocesspd cpu的加在一起
    mergeprocesspd["cpu"] = mergeprocesspd["usr_cpu"] + mergeprocesspd["kernel_cpu"]
    iserverpd["cpu"] = iserverpd["usr_cpu"] + iserverpd["kernel_cpu"]
    sub_server_process_cpu = iserverpd["cpu"] - mergeprocesspd["cpu"]
    # 如果iserverpd["cpu"]是0， 就当作1
    iserverpd["cpu"] = iserverpd["cpu"].apply(lambda x: 1 if x == 0 else x)
    cpu_change = sub_server_process_cpu / iserverpd["cpu"]


    # 返回结果
    respd = pd.DataFrame()
    respd[TIME_COLUMN_NAME] = iserverpd[TIME_COLUMN_NAME]
    # ========
    respd["cpu_change"] = cpu_change
    respd["cpu_change_smooth"] = smoothseries(cpu_change)
    respd["ddrc_ddwr_sum"] = itopdownpd["ddrc_ddwr_sum_median"]
    respd["ddrc_ddwr_sum_compensation"] = itopdownpd["ddrc_ddwr_sum_median"] * (1 + cpu_change)
    # respd["mflops"] = itopdownpd["mflops"]
    # respd["mflops_change"] = mflops_change
    # respd["mflops_mean"] = mflops_mean
    respd["ddrc_ddwr_sum_compensation_mflops"] = itopdownpd["ddrc_ddwr_sum_median"] * (1 + mflops_change)
    respd["ddrc_ddwr_sum_compensation_mflops_smooth"] = smoothseries(respd["ddrc_ddwr_sum_compensation_mflops"])
    # =======
    respd[FAULTFLAG] = iserverpd[FAULTFLAG]
    return respd



def gettitle(ipath: str):
    B, C = os.path.split(ipath)
    A, B = os.path.split(B)
    return "{}/{}".format(B, C)

# 查看processcpu时间进行补偿的数据
if __name__ == "__main__":
    dirpathes = [
        R"csvfiles/abnormals/allcpu10/1/hpc_topdown.csv",
    ]
    for dirpath in dirpathes:
        title = gettitle(dirpath)

        iserverpd, itopdownpd, iprocesspd = getServerTopdownandProcesspds(dirpath)
        iserverpd, itopdownpd, iprocesspd = getsametimepdList([iserverpd, itopdownpd, iprocesspd])
        respd = getSUMWR(itopdownpd, iprocesspd, iserverpd)

        respd.set_index("time", inplace=True)
        n_cols_plot(respd, respd.columns, title)
