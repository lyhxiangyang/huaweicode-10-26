import os
from typing import List

import pandas as pd

from hpc.l3l2utils.DefineData import TIME_COLUMN_NAME, PID_FEATURE, CPU_FEATURE, FAULT_FLAG

"""
将[dataframe1, dataframe2, dataframe3] 这种结构进行存储 name为0.csv 1.csv 2.csv
"""


def saveDFListToFiles(spath: str, pds: List[pd.DataFrame]):
    if not os.path.exists(spath):
        os.makedirs(spath)
    for i in range(0, len(pds)):
        savefilepath = os.path.join(spath, str(i) + ".csv")
        pds[i].to_csv(savefilepath, index=False)


def getServer_Process_l2_Network_Ping_TopdownList(dirpath: str, server_feature=None, process_feature=None, l2_feature=None,
                                          ping_feature=None,
                                          network_feature=None, topdown_feature=None,isExistFlag: bool = True):
    def getfilespath(filepath: str) -> List[str]:
        if not os.path.exists(filepath):
            print("{}路径不存在".format(filepath))
            exit(1)
        files = os.listdir(filepath)
        filepaths = [os.path.join(filepath, i) for i in files]
        return filepaths

    serverfiles = getfilespath(os.path.join(dirpath, "server"))
    processfiles = getfilespath(os.path.join(dirpath, "process"))
    l2files = getfilespath(os.path.join(dirpath, "compute"))
    networkfiles = getfilespath(os.path.join(dirpath, "nic"))
    pingfiles = getfilespath(os.path.join(dirpath, "ping"))
    topdownfiles = getfilespath(os.path.join(dirpath, "topdown"))
    if server_feature is not None:
        time_server_feature = server_feature.copy()
        time_server_feature.extend([TIME_COLUMN_NAME])
    if process_feature is not None:
        time_process_feature = process_feature.copy()
        time_process_feature.extend([TIME_COLUMN_NAME, PID_FEATURE, CPU_FEATURE])
    if l2_feature is not None:
        time_l2_feature = l2_feature.copy()
        time_l2_feature.extend([TIME_COLUMN_NAME])
    if network_feature is not None:
        time_network_feature = network_feature.copy()
        time_network_feature.extend([TIME_COLUMN_NAME])
    if ping_feature is not None:
        time_ping_feature = ping_feature.copy()
        time_ping_feature.extend([TIME_COLUMN_NAME])
    if topdown_feature is not None:
        time_topdown_feature = topdown_feature.copy()
        time_topdown_feature.extend([TIME_COLUMN_NAME])

    if isExistFlag:
        if server_feature is not None:
            time_server_feature.extend([FAULT_FLAG])
        if process_feature is not None:
            time_process_feature.extend([FAULT_FLAG])
        if l2_feature is not None:
            time_l2_feature.extend([FAULT_FLAG])
        if network_feature is not None:
            time_network_feature.extend([FAULT_FLAG])
        if ping_feature is not None:
            time_ping_feature.extend([FAULT_FLAG])
        if topdown_feature is not None:
            time_topdown_feature.extend([FAULT_FLAG])

    processpds = []
    serverpds = []
    l2pds = []
    networkpds = []
    pingpds = []
    topdownpds = []
    # 预测进程数据
    for ifile in processfiles:
        tpd = getfilepd(ifile)
        if process_feature is not None:
            tpd = tpd.loc[:, time_process_feature]
        processpds.append(tpd)
    # 预测服务数据
    for ifile in serverfiles:
        tpd = getfilepd(ifile)
        if server_feature is not None:
            tpd = tpd.loc[:, time_server_feature]
        serverpds.append(tpd)
    # 预测l2数据
    for ifile in l2files:
        tpd = getfilepd(ifile)
        if l2_feature is not None:
            tpd = tpd.loc[:, time_l2_feature]
        l2pds.append(tpd)
    # 预测网络数据
    for ifile in networkfiles:
        tpd = getfilepd(ifile)
        if network_feature is not None:
            tpd = tpd.loc[:, time_network_feature]
        networkpds.append(tpd)
    for ifile in pingfiles:
        tpd = getfilepd(ifile)
        if ping_feature is not None:
            tpd = tpd.loc[:, time_ping_feature]
        pingpds.append(tpd)
    for ifile in topdownfiles:
        tpd = getfilepd(ifile)
        if topdown_feature is not None:
            tpd = tpd.loc[:, time_topdown_feature]
        topdownpds.append(tpd)
    return serverpds, processpds, l2pds, networkpds, pingpds, topdownpds


"""
读一个文件 
如果ifile为None表示ipath本身就表示一个文件的名字
如果ifile不为None表示ipath只是一个路径， ifile表示此路径下
如果features为空
"""


def getfilepd(ipath: str, ifile: str = None, features: List[str] = None) -> pd.DataFrame:
    if not os.path.exists(ipath):
        filename = os.path.basename(ipath)
        print("{} 文件不存在".format(filename))
        exit(1)
    if ifile is not None:
        ipath = os.path.join(ipath, ifile)
    tpd = pd.read_csv(ipath)
    if features is not None:
        return tpd[:, features]
    return tpd


"""
保存一个文件
"""


def savepdfile(ds, spath, filename, index: bool = False):
    if not os.path.exists(spath):
        os.makedirs(spath)
    pathfilename = os.path.join(spath, filename)
    ds.to_csv(pathfilename, index=index)
