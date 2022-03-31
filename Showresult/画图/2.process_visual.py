import os
import time
from typing import List

import pandas as pd
import plotly.graph_objs as go

from hpc.l3l2utils.FeatureExtraction import differenceProcess


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


TIMELABLE = "time"
FAULTFLAG = "faultFlag"
PID_FEATURE = "pid"

'''
## 参数解释
# nowtime是一个数字，表示当前时间，是一个大数字
## 返回值解释
# 返回的是一个字符串形式的数字, 格式为"2021-05-12T18:19:36+00:00"
'''


def ChangeTimeToStr(nowtime: int) -> str:
    struct_time = time.localtime(nowtime)
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", struct_time)


'''
## 参数解释
# timestamp是一个字符串的表示时间的形式，形式严格如："2021-05-12T18:19:36+00:00"
## 返回值解释
# 返回的是一个数字，表示当前的时间
'''


def ChangeStrToTime(timestamp: str) -> float:
    struct_time = time.strptime(timestamp, "%Y-%m-%dT%H:%M:%S+00:00")
    return time.mktime(struct_time)


"""
从time.md中得到开始时间和结束时间
"""


def GetStartEndTimeFromTimeMd(timemdpath: str) -> List[float]:
    res = ["", ""]
    with open(timemdpath, "r") as f:
        readLines = f.readlines()
    readLines = [i.strip() for i in readLines if i != '\n']
    if len(readLines) > 0:
        res[0] = ChangeStrToTime(readLines[0])
    if len(readLines) > 1:
        res[1] = ChangeStrToTime(readLines[1])
    return res


"""
给表格打上标签
df的index是时间
"""


def setpdFlag(df: pd.DataFrame, flag: int, beginTime: float, endTime: float):
    def judgetime(nowtime: str) -> bool:
        nowIntTime = ChangeStrToTime(nowtime)
        if beginTime <= nowIntTime <= endTime:
            return True
        return False

    if FAULTFLAG not in df.columns.array:
        df[FAULTFLAG] = [-1] * len(df)
    isConform = df.index.to_series().apply(judgetime)
    df.loc[isConform, FAULTFLAG] = [flag] * len(df[isConform])


def processing(filepath: str, filename: str = None):
    file = filepath
    if filename is not None:
        file = os.path.join(filepath, filename)
    df = pd.read_csv(file, index_col=TIMELABLE)
    df: pd.DataFrame
    if "faultFlag" not in df.columns:
        df["faultFlag"] = 0
    df = df.dropna()
    # 修改列名 去掉每个文件中的空格
    df.rename(columns=lambda x: x.replace('\g', '').strip(), inplace=True)
    if "User" in df.columns and "System" in df.columns:
        df["CPU"] = df["User"] + df["System"]
    df = df.copy()
    df['flag'] = df['faultFlag'].apply(lambda x: x % 10)
    df = df.dropna()
    return df


def processingpd(processpd: pd.DataFrame):
    df = processpd.set_index(TIMELABLE)
    if "faultFlag" not in df.columns:
        df["faultFlag"] = 0
    df = df.dropna()
    # 修改列名 去掉每个文件中的空格
    df.rename(columns=lambda x: x.replace('\g', '').strip(), inplace=True)
    if "User" in df.columns and "System" in df.columns:
        df["CPU"] = df["User"] + df["System"]
    df = df.copy()
    df['flag'] = df['faultFlag'].apply(lambda x: x % 10)
    df = df.dropna()
    return df


'''
得到pid的cpu信息
'''


def getpidcpuInfo(processpd: pd.DataFrame):
    respd = pd.DataFrame()
    for icore, icorepd in processpd.groupby("cpu_affinity"):
        icorepd = icorepd.reset_index(drop=True)
        cname = "core{}_cpu".format(icore)
        cpuSeries = icorepd["user"] + icorepd["system"]
        respd[cname] = cpuSeries
        if FAULTFLAG not in respd.columns:
            respd[FAULTFLAG] = icorepd[FAULTFLAG]
        if TIMELABLE not in respd.columns:
            respd[TIMELABLE] = icorepd[TIMELABLE]
    return respd


"""
合并所有的process数据信息
"""


def mergeProceeDF(processpd: pd.DataFrame, sumFeatures=None):
    if sumFeatures is None:
        sumFeatures = ["time", "usr_cpu", "kernel_cpu", "memory_percent"]
    respd = pd.DataFrame()
    tpd = processpd[sumFeatures].groupby("time").sum()
    return tpd


if __name__ == "__main__":
    dirpathes = [
        R"csvfiles/5.3组正常数据-1min/1/hpc_process.csv",
    ]
    for dirpath in dirpathes:
        if not dirpath.strip().endswith(".csv"):
            continue
        normal_file_name = dirpath.strip()
        df = pd.read_csv(normal_file_name)
        if FAULTFLAG not in df.columns:
            df[FAULTFLAG] = 0

        df = differenceProcess([df], accumulateFeatures=["usr_cpu", "kernel_cpu"])[0]
        dfsum = mergeProceeDF(df)
        dfinfo = getpidcpuInfo(df)
        dfinfo = processingpd(dfinfo)
        n_cols_plot(dfinfo, dfinfo.columns, normal_file_name)
