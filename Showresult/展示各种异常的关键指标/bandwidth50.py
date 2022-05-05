# import os
# import time
# from typing import List
#
# import pandas as pd
# import plotly.graph_objs as go
#
# from Showresult.显示process数据.processmemory import FAULTFLAG
# from hpc.l3l2utils.DataFrameOperation import mergeinnerTwoDataFrame, smoothseries
# from hpc.l3l2utils.DataOperation import changeTimeToFromPdlists, getRunHPCTimepdsFromProcess, getsametimepdList
# from hpc.l3l2utils.DefineData import TIME_COLUMN_NAME
# from hpc.l3l2utils.FeatureExtraction import differenceServer, differenceProcess
#
#
# def n_cols_plot(df, yy, title):
#     data = []
#     for y in yy:
#         trace1 = go.Scatter(x=df.index, y=df[y], mode='lines', name=y)
#         data.append(trace1)
#         # abnormal_point = df[(df["faultFlag"] != 0)]
#         abnormal_point = df[df["faultFlag"].apply(lambda x: True if x != 0 and x != -1 else False)]
#         trace = go.Scatter(x=abnormal_point.index, y=abnormal_point[y], mode='markers')
#         data.append(trace)
#     layout = dict(title=title)
#     fig = go.Figure(data=data, layout=layout)
#     fig.show()
#
#
# def processing(filepath: str, filename: str = None):
#     file = filepath
#     if filename is not None:
#         file = os.path.join(filepath, filename)
#     itopdownpd = pd.read_csv(file, index_col=TIME_COLUMN_NAME)
#     if FAULTFLAG not in itopdownpd.columns:
#         itopdownpd[FAULTFLAG] = 0
#     itopdownpd = itopdownpd.dropna()
#     itopdownpd['flag'] = itopdownpd['faultFlag'].apply(lambda x: x % 10)
#     itopdownpd = itopdownpd.dropna()
#     return itopdownpd
#
#
# # 得到server和process的pd
# def getServerTopdownandProcesspds(filepath: str):
#     itopdownpath = os.path.join(filepath, "topdown", "topdown.csv")
#     iprocesspath = os.path.join(filepath, "process", "hpc_process.csv")
#     iserverpath = os.path.join(filepath, "server", "metric_server.csv")
#
#     # 读取到dataframe中
#     itopdownpd = pd.read_csv(itopdownpath)
#     iprocesspd = pd.read_csv(iprocesspath)
#     iserverpd = pd.read_csv(iserverpath)
#
#     # 对iserver进行时间处理
#     serverlists = changeTimeToFromPdlists([iserverpd], isremoveDuplicate=True)
#     topdownlists = changeTimeToFromPdlists([itopdownpd], isremoveDuplicate=True)
#     processpdlists = changeTimeToFromPdlists([iprocesspd])
#     # 对数据进行差分处理
#     serverlists = differenceServer(serverlists, ["pgfree", "usr_cpu", "kernel_cpu"])
#     # topdownlists = differenceServer(topdownlists, ["pgfree", "usr_cpu", "kernel_cpu"])
#     processpdlists = differenceProcess(processpdlists, ["usr_cpu", "kernel_cpu"])
#     return serverlists[0], topdownlists[0], processpdlists[0]
#
#
#
# def gettitle(ipath: str):
#     B, C = os.path.split(ipath)
#     A, B = os.path.split(B)
#     return "{}/{}".format(B, C)
#
#
# # 查看processcpu时间进行补偿的数据
# if __name__ == "__main__":
#     dirpathes = [
#         R"csvfiles/abnormals/allcpu10/centos11",
#     ]
#     for dirpath in dirpathes:
#         title = gettitle(dirpath)
#
#         iserverpd, itopdownpd, iprocesspd = getServerTopdownandProcesspds(dirpath)
#         iserverpd, itopdownpd, iprocesspd = getsametimepdList([iserverpd, itopdownpd, iprocesspd])
#
#         respd = pass
#
#         respd.set_index("time", inplace=True)
#         n_cols_plot(respd, respd.columns, title)
