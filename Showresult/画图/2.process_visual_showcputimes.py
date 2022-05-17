# coding=utf-8
import os
import time
from typing import List

import pandas as pd
import plotly.graph_objs as go

from hpc.l3l2utils.DataFrameOperation import smoothseries
from hpc.l3l2utils.DataOperation import changeTimeToFromPdlists
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
# timestamp是一个字符串的表示时间的形式，形式严格如："2021-05-12T18:19:36+00:00"
## 返回值解释
# 返回的是一个数字，表示当前的时间
'''






def processingpd(processpd: pd.DataFrame):
    df = processpd.set_index(TIMELABLE)
    if "faultFlag" not in df.columns:
        df["faultFlag"] = 0
    df = df.dropna()
    # 修改列名 去掉每个文件中的空格
    if "usr_cpu" in df.columns and "kernel_cpu" in df.columns:
        df["CPU"] = df["usr_cpu"] + df["kernel_cpu"]
    df = df.copy()
    df['flag'] = df['faultFlag'].apply(lambda x: x % 10)
    df = df.dropna()
    return df


'''
得到pid的cpu信息
'''


def getpidcpuInfo(processpd: pd.DataFrame):
    respd = pd.DataFrame(index=processpd["time"].drop_duplicates())
    processpd.set_index("time", inplace=True)
    for icore, icorepd in processpd.groupby("cpu_affinity"):
        # icorepd = icorepd.reset_index(drop=True)
        # 去重
        icorepd = icorepd[~icorepd.index.duplicated(keep="first")]
        cname = "core{}_cpu".format(icore)
        cpuSeries = pd.Series(data=icorepd["cpu"], name=cname)
        respd = pd.concat([respd, cpuSeries], axis=1)
        if FAULTFLAG not in respd.columns.tolist():
            respd[FAULTFLAG] = icorepd[FAULTFLAG]
    respd.fillna(-1, inplace=True)
    respd.reset_index(drop=False, inplace=True)
    return respd

"""
合并所有的process数据信息
"""


def mergeProceeDF(processpd: pd.DataFrame, sumFeatures=None):
    if sumFeatures is None:
        sumFeatures = ["time", "usr_cpu", "kernel_cpu", "mem_percent"]
    tpd = processpd[sumFeatures].groupby("time").sum()
    return tpd


if __name__ == "__main__":
    dirpathes = [
        R"csvfiles/abnormals_10s_33min注入/allcpu10/1/hpc_process.csv",
    ]
    for dirpath in dirpathes:
        if not dirpath.strip().endswith(".csv"):
            continue
        normal_file_name = dirpath.strip()
        df = pd.read_csv(normal_file_name)
        if FAULTFLAG not in df.columns:
            df[FAULTFLAG] = 0
        df = changeTimeToFromPdlists([df])[0]
        df = differenceProcess([df], accumulateFeatures=["usr_cpu", "kernel_cpu"])[0]
        # dfsum = mergeProceeDF(df)
        dfinfo = getpidcpuInfo(df)
        dfinfo = processingpd(dfinfo)
        dfinfo["cputime55"] = 55
        n_cols_plot(dfinfo, dfinfo.columns, normal_file_name)
