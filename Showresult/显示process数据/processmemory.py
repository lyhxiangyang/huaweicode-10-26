import os
import plotly.graph_objs as go
import os.path
import pandas as pd
# 对process文件进行处理 主要是内存数据，然后联合server数据进行合并
from hpc.l3l2utils.DataOperation import getsametimepd, changeTimeToFromPdlists
from hpc.l3l2utils.DefineData import TIME_COLUMN_NAME
from hpc.l3l2utils.FeatureExtraction import differenceServer, differenceProcess

TIMELABLE = "Timestamp"
FAULTFLAG = "faultFlag"

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


def processing1(filepath: str, filename: str):
    file = os.path.join(filepath, filename)
    df = pd.read_csv(file, index_col=TIMELABLE)
    df = df.dropna()
    # 修改列名 去掉每个文件中的空格
    df.rename(columns=lambda x: x.replace('\g', '').strip(), inplace=True)
    if "User" in df.columns and "System" in df.columns:
        df["CPU"] = df["User"] + df["System"]
    df = df.copy()
    df['flag'] = df['faultFlag'].apply(lambda x: x % 10)
    df = df.dropna()
    return df


############################################################################################################ 进程数据处理

def mergeProceeDF(processpd: pd.DataFrame, sumFeatures=None):
    if sumFeatures is None:
        sumFeatures = [TIME_COLUMN_NAME, "usr_cpu", "kernel_cpu", "mem_percent"]
    if TIME_COLUMN_NAME not in sumFeatures:
        sumFeatures.append(TIME_COLUMN_NAME)
    tpd = processpd[sumFeatures].groupby("time").sum()
    tpd.reset_index(drop=False, inplace=True)
    tpd.reset_index(drop=True, inplace=True)
    return tpd

# 传入进去的process应该是相同时间的
# 根据server总内存和process mempercent来得到数据
def subtractionMemory(serverpd: pd.DataFrame, processpd: pd.DataFrame) -> pd.Series:
    # 保证serverpd和processpd的时间变化范围是一致的
    sametimeserverpd, sametimeprocesspd = getsametimepd(serverpd, processpd)
    assert len(sametimeserverpd) == len(sametimeprocesspd)

    allservermemory = serverpd["mem_total"].iloc[0]
    sametimeserverpd["processtime"] = sametimeprocesspd[TIME_COLUMN_NAME]
    sametimeserverpd["processmemory"] = sametimeprocesspd["mem_percent"] * allservermemory
    sametimeserverpd["othermemory"] = sametimeserverpd["mem_used"] - sametimeserverpd["processmemory"]
    return sametimeserverpd

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
    processpdlists = differenceProcess(processpdlists, ["usr_cpu", "kernel_cpu"])

    iprocesspd = mergeProceeDF(processpdlists[0])
    return serverpdlists[0], iprocesspd



if __name__ == "__main__":
    # normal_file_name = "E5_1KM_训练"
    # filepath = R"DATA/test1.csv"
    showfiles = [
        "network.csv",
        "cpu.csv",
        "memory.csv",
        "load.csv",
    ]

    dirpathes = [
        R"/Users/liyanghan/Desktop/Ganglia数据整理/dial数据/Dial_intensity_1/dial_1_19"
    ]
    for dirpath in dirpathes:
        for filename in os.listdir(dirpath):
            if not filename.strip().endswith(".csv"):
                continue
            if filename not in showfiles:
                continue
            normal_file_name = filename.strip()
            filepath = os.path.join(dirpath, normal_file_name)
            df = processing1(dirpath, normal_file_name)
            n_cols_plot(df, df.columns, normal_file_name)
