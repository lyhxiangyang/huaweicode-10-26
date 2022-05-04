import os
import time
from typing import List

import pandas as pd
import plotly.graph_objs as go

from hpc.l3l2utils.DataFrameSaveRead import getfilepd
from hpc.l3l2utils.DataOperation import changeTimeToFromPdlists, getsametimepd
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
    df = pd.read_csv(file, index_col="time")
    df: pd.DataFrame
    if "faultFlag" not in df.columns:
        df["faultFlag"] = 0
    df = df.dropna()
    # 修改列名 去掉每个文件中的空格
    df = df.copy()
    df['flag'] = df['faultFlag'].apply(lambda x: x % 10)
    df = df.dropna()
    return df

def processingpd(df: pd.DataFrame):
    if "time" in df.columns.array:
        df.set_index("time", inplace=True)
    if "faultFlag" not in df.columns:
        df["faultFlag"] = 0
    # 修改列名 去掉每个文件中的空格
    df = df.copy()
    df['flag'] = df['faultFlag'].apply(lambda x: x % 10)
    df = df.dropna()
    return df


# ======================================================== 上面是画图函数
# ======================================================== 下面是处理函数

def mergeProceeDF(processpd: pd.DataFrame, sumFeatures=None):
    if sumFeatures is None:
        sumFeatures = [TIME_COLUMN_NAME, "usr_cpu", "kernel_cpu", "mem_percent", "rss", "vms"]
    if TIME_COLUMN_NAME not in sumFeatures:
        sumFeatures.append(TIME_COLUMN_NAME)
    tpd = processpd[sumFeatures].groupby("time").sum()
    tpd.reset_index(drop=False, inplace=True)
    return tpd





# 得到server和process的pd
def getserverandprocesspds(filepath: str):
    iserverpath = os.path.join(filepath, "server", "metric_server.csv")
    iprocesspath = os.path.join(filepath, "process", "hpc_process.csv")

    # 读取到dataframe中
    iserverpd = pd.read_csv(iserverpath)
    iprocesspd = pd.read_csv(iprocesspath)

    # 对iserver进行时间处理
    serverpdlists = changeTimeToFromPdlists([iserverpd], isremoveDuplicate=True)
    processpdlists = changeTimeToFromPdlists([iprocesspd])
    # 对数据进行差分处理
    serverpdlists = differenceServer(serverpdlists, ["pgfree"])

    # 画出pgfree的平均数
    for ipd in serverpdlists:
        ipd["pgfree_mean"] = getSeriesFrequencyMean(ipd["pgfree"])


    processpdlists = differenceProcess(processpdlists, ["usr_cpu", "kernel_cpu"])
    iprocesspd = mergeProceeDF(processpdlists[0])
    # 得到相同时间段
    a,b = getsametimepd(serverpdlists[0], iprocesspd)
    return a,b

# 传入进去的process应该是相同时间的
# 根据server总内存和process mempercent来得到数据
def subtractionMemory(serverpd: pd.DataFrame, processpd: pd.DataFrame) -> pd.DataFrame:
    # 保证serverpd和processpd的时间变化范围是一致的
    # sametimeserverpd, sametimeprocesspd = getsametimepd(serverpd, processpd)
    sametimeserverpd, sametimeprocesspd = serverpd, processpd
    assert len(sametimeserverpd) == len(sametimeprocesspd)

    sametimeserverpd["processtime"] = sametimeprocesspd[TIME_COLUMN_NAME]
    sametimeserverpd["s_used"] = sametimeserverpd["mem_total"] - sametimeserverpd['mem_avail']
    sametimeserverpd["p_rss"] = sametimeprocesspd["rss"]
    sametimeserverpd["p_vms"] = sametimeprocesspd["vms"]
    sametimeserverpd["p_used-rss"] = sametimeserverpd["mem_used"] - sametimeprocesspd["rss"]

    return sametimeserverpd

def gettitle(ipath: str):
    B,C=os.path.split(ipath)
    A,B=os.path.split(B)
    return "{}/{}".format(B,C)


# 将数组划分为10min，然后取数值最大的
def getSeriesFrequencyMean(dataseries: pd.Series, bins=10):
    # 先划分成10分
    tpd = pd.DataFrame(data={
        "origindata": dataseries
    })
    tpd["cutdata"] = pd.cut(tpd["origindata"], bins=bins)
    # 得到最大值对应的索引，也就是分组
    maxvaluecut=pd.value_counts(tpd["cutdata"]).idxmax()
    meanvalues = tpd.groupby("cutdata").get_group(maxvaluecut)["origindata"].mean()
    return meanvalues


if __name__ == "__main__":
    dirpathes = [
        R"DATA/2022-01-14新的测试数据/1.wrf_1km_multi_l3/centos11",
    ]
    for dirpath in dirpathes:
        title=gettitle(dirpath)

        ipd = getfilepd(dirpath)
        # 画出来
        ipd = processingpd(ipd)
        n_cols_plot(ipd,ipd.columns,title)




