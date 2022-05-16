import os
import time
from typing import List

import pandas as pd
import plotly.graph_objs as go

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
    df = differenceServer([df], ["pgfree", "usr_cpu", "kernel_cpu"])[0]
    df["cpu"] = df["usr_cpu"] + df["kernel_cpu"]
    df = df.dropna()
    # 修改列名 去掉每个文件中的空格
    df = df.copy()
    df['flag'] = df['faultFlag'].apply(lambda x: x % 10)
    df = df.dropna()
    return df

def processingpd(df: pd.DataFrame):
    if "time" in df.columns.array:
        df.set_index("time", inplace=True)
    if "faultFlag" not in df.columngs:
        df["faultFlag"] = 0
    # 修改列名 去掉每个文件中的空格
    df["pgfree"] = differenceServer([df], ["pgfree"])[0]
    df = df.copy()
    df['flag'] = df['faultFlag'].apply(lambda x: x % 10)
    df = df.dropna()
    return df


# ======================================================== 上面是画图函数
# ======================================================== 下面是处理函数


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

        serverpd = processing(dirpath)
        # 画出来
        n_cols_plot(serverpd,serverpd.columns,title)




