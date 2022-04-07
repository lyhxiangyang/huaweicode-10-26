import os
import time
from typing import List

import pandas as pd
import plotly.graph_objs as go

from hpc.l3l2utils.DataFrameOperation import mergeinnerTwoDataFrame
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

def mergeProceeDF(processpd: pd.DataFrame, sumFeatures=None, inplace=True):
    if sumFeatures is None:
        return pd.DataFrame()
    if inplace:
        processpd = processpd.copy()
    if TIME_COLUMN_NAME not in sumFeatures:
        sumFeatures.append(TIME_COLUMN_NAME)
    tpd = processpd[sumFeatures].groupby("time").sum()
    tpd1 = processpd[[TIME_COLUMN_NAME, "pid"]].groupby("time").first()
    tpd.reset_index(drop=False, inplace=True)
    tpd1.reset_index(drop=False, inplace=True)
    respd = mergeinnerTwoDataFrame(lpd=tpd, rpd=tpd1)
    return respd


# 得到server和process的pd
def getserverandprocesspds(filepath: str):
    iserverpath = os.path.join(filepath, "hpc_server.csv")
    iprocesspath = os.path.join(filepath, "hpc_process.csv")

    # 读取到dataframe中
    iserverpd = pd.read_csv(iserverpath)
    iprocesspd = pd.read_csv(iprocesspath)

    # 对iserver进行时间处理
    serverpdlists = changeTimeToFromPdlists([iserverpd], isremoveDuplicate=True)
    processpdlists = changeTimeToFromPdlists([iprocesspd])
    # 对数据进行差分处理
    serverpdlists = differenceServer(serverpdlists, ["pgfree"])
    processpdlists = differenceProcess(processpdlists, ["usr_cpu", "kernel_cpu"])

    return serverpdlists[0], processpdlists[0]

    # iprocesspd = mergeProceeDF(processpdlists[0])
    # # 得到相同时间段
    # # a,b = getsametimepd(serverpdlists[0], iprocesspd)
    # pspd = pd.merge(left=serverpdlists[0], right=iprocesspd, left_on="time", right_on="time", how="left",
    #                 suffixes=("", "_y"))
    # pspd.fillna(-1, inplace=True)
    # return pspd

def smoothseries(cseries: pd.Series, windows=5)->pd.Series:
    mediansmooth = cseries.rolling(window=5, min_periods=1, center=True).median()
    meanmediansmooth = mediansmooth.rolling(window=windows, min_periods=1, center=True).mean()
    return meanmediansmooth
def mediansmoothseries(cseries: pd.Series, windows=5)->pd.Series:
    mediansmooth = cseries.rolling(window=windows, min_periods=1, center=True).median()
    return mediansmooth
def meansmoothseries(cseries: pd.Series, windows=5)->pd.Series:
    meanmediansmooth = cseries.rolling(window=windows, min_periods=1, center=True).mean()
    return meanmediansmooth

def maxmoothseries(cseries: pd.Series)->pd.Series:
    meanmediansmooth = cseries.rolling(window=5, min_periods=1, center=True).max()
    return meanmediansmooth



# 对内存的差值处理必须根据进程pid号进行处理
# 保证内存的指标和pid号的指标长度是一样的
def diffmemoryseries(memseries: pd.Series, pidseries: pd.Series):
    assert len(memseries) == len(pidseries)
    df = pd.DataFrame(data={
        "mem": memseries,
        "pid": pidseries
    })
    # 直接写个for循环
    reslists = []
    winsize = 5
    for ipid, idf in df.groupby("pid"):
        other_mem_smooth = smoothseries(idf["mem"], windows=winsize)
        other_mem_smooth_diff = other_mem_smooth.diff(1).fillna(0)
        reslists.extend(other_mem_smooth_diff.tolist())

    return pd.Series(data=reslists)




# 传入进去的process应该是相同时间的
# 根据server总内存和process mempercent来得到数据
def subtractionMemory(serverpd: pd.DataFrame, processpd: pd.DataFrame) -> pd.DataFrame:
    mergeprocesspd = mergeProceeDF(processpd, sumFeatures=["rss"])
    # 将两者合并
    pspd = pd.merge(left=serverpd, right=mergeprocesspd, left_on=TIME_COLUMN_NAME, right_on=TIME_COLUMN_NAME,
                    how="left", suffixes=("", "_y"))
    pspd.fillna(0, inplace=True)  # 认为进程不在的时候其数据为0

    servermem = pspd["mem_used"]
    processmem = pspd["rss"]
    othermem = servermem - processmem
    pspd["s-p-mem"] = othermem

    othermemdiff = diffmemoryseries(othermem, pspd["pid"])
    pspd["s-p-mem-diff"] = othermemdiff

    return pspd


def gettitle(ipath: str):
    B, C = os.path.split(ipath)
    A, B = os.path.split(B)
    return "{}/{}".format(B, C)


if __name__ == "__main__":
    dirpathes = [
        R"csvfiles/abnormals/memleak60/1",
        R"csvfiles/abnormals/memleak60/2",
        # R"csvfiles/5.3组正常数据-1min/2",
    ]
    for dirpath in dirpathes:
        title = gettitle(dirpath)

        serverpd, processpd = getserverandprocesspds(dirpath)
        serverpd = subtractionMemory(serverpd, processpd)

        # 画出来
        processingpd(serverpd)
        n_cols_plot(serverpd, serverpd.columns, title)

