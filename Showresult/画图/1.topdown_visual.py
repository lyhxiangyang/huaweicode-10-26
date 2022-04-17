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
    df = pd.read_csv(file, index_col=TIME_COLUMN_NAME)
    if FAULTFLAG not in df.columns:
        df[FAULTFLAG] = 0
    df = df.dropna()
    # 修改列名 去掉每个文件中的空格
    df["mflops_median"] = df["mflops"].rolling(window=5, center=True, min_periods=1).median()
    df["mflops_median_mean"] = df["mflops"].rolling(window=5, center=True, min_periods=1).mean()
    df = df.copy()
    df['flag'] = df['faultFlag'].apply(lambda x: x % 10)
    df = df.dropna()
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
