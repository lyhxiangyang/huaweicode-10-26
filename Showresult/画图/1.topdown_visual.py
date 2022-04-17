import os
import time
from typing import List

import pandas as pd
import plotly.graph_objs as go

from Showresult.显示process数据.processmemory import FAULTFLAG
from hpc.l3l2utils.DefineData import TIME_COLUMN_NAME


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
    return df


if __name__ == "__main__":
    dirpathes = [
        R"csvfiles/abnormals/allcpu10/1/hpc_topdown.csv",
    ]
    for dirpath in dirpathes:
        if not dirpath.strip().endswith(".csv"):
            continue
        normal_file_name = dirpath.strip()
        df = processing(normal_file_name)
        # df = pd.concat([x, y], axis=1)
        n_cols_plot(df, df.columns, normal_file_name)
