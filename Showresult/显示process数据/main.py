import os.path

import pandas as pd
# 对process文件进行处理 主要是内存数据，然后联合server数据进行合并
from hpc.l3l2utils.DataOperation import getsametimepd, changeTimeToFromPdlists
from hpc.l3l2utils.DefineData import TIME_COLUMN_NAME
from hpc.l3l2utils.FeatureExtraction import differenceServer, differenceProcess


def mergeProceeDF(processpd: pd.DataFrame, sumFeatures=None):
    if sumFeatures is None:
        sumFeatures = [TIME_COLUMN_NAME, "usr_cpu", "kernel_cpu", "mem_percent"]
    if TIME_COLUMN_NAME not in sumFeatures:
        sumFeatures.append(TIME_COLUMN_NAME)
    tpd = processpd[sumFeatures].groupby("time").sum()
    tpd.reset_index(drop=False, inplace=True)
    tpd.reset_index(drop=True, inplace=True)
    return tpd

def subtractionMemory(serverpd: pd.DataFrame, processpd: pd.DataFrame) -> pd.DataFrame:
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
    main.py





































































































