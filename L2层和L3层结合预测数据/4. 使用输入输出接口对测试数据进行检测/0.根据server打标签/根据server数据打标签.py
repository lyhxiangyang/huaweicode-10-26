import os
from typing import List, Dict

import pandas as pd

from l3l2utils.DataFrameSaveRead import getfilepd, savepdfile
from l3l2utils.DataOperation import changeTimeFromOnepd
from l3l2utils.DefineData import TIME_COLUMN_NAME, FAULT_FLAG

"""
这个文件用来根据server数据的时间对其他文件打上标签
"""

F = {
    "server": "metric_server.csv",
    "process": "hpc_process.csv",
    "l2": "compute.csv",
    "network": "nic.csv",
    "ping": "ping.csv",
    "topdown": "topdown.csv",
}
addFlagDir = [
    R"DATA/2022-01-14新的测试数据/23.复合错误wrf_1km_multi_node_single_50cpu_309/centos11-flag",
    R"DATA/2022-01-14新的测试数据/23.复合错误wrf_1km_multi_node_single_50cpu_309/centos16-flag",
    R"DATA/2022-01-14新的测试数据/23.复合错误wrf_1km_multi_node_single_50cpu_309/centos21",
    R"DATA/2022-01-14新的测试数据/23.复合错误wrf_1km_multi_node_single_50cpu_309/centos26",

    R"DATA/2022-01-14新的测试数据/24.复合错误wrf_1km_single_50cpu_server_power_capping_310/centos11",
    R"DATA/2022-01-14新的测试数据/24.复合错误wrf_1km_single_50cpu_server_power_capping_310/centos16-flag",
    R"DATA/2022-01-14新的测试数据/24.复合错误wrf_1km_single_50cpu_server_power_capping_310/centos21",
    R"DATA/2022-01-14新的测试数据/24.复合错误wrf_1km_single_50cpu_server_power_capping_310/centos26",

]


def getDirs(dirpaths) -> List[str]:
    dirnamess = os.listdir(dirpaths)
    dirlists = [os.path.join(dirpaths, idir, "centos11") for idir in dirnamess if os.path.exists(os.path.join(dirpaths, idir, "centos11"))]
    dirlists.extend([os.path.join(dirpaths, idir, "centos16") for idir in dirnamess if os.path.exists(os.path.join(dirpaths, idir, "centos16"))])
    dirlists.extend([os.path.join(dirpaths, idir, "centos21") for idir in dirnamess if os.path.exists(os.path.join(dirpaths, idir, "centos21"))])
    dirlists.extend([os.path.join(dirpaths, idir, "centos26") for idir in dirnamess if os.path.exists(os.path.join(dirpaths, idir, "centos26"))])
    return dirlists


def getOneDirPd(dirpath: str):
    resDict = {}
    resDict["server"] = changeTimeFromOnepd(getfilepd(dirpath, F["server"]))
    resDict["process"] = changeTimeFromOnepd(getfilepd(dirpath, F["process"]))
    resDict["l2"] = changeTimeFromOnepd(getfilepd(dirpath, F["l2"]))
    resDict["network"] = changeTimeFromOnepd(getfilepd(dirpath, F["network"]))
    resDict["ping"] = changeTimeFromOnepd(getfilepd(dirpath, F["ping"]))
    resDict["topdown"] = changeTimeFromOnepd(getfilepd(dirpath, F["topdown"]))
    return resDict


"""
根据server文件中的时间和addflag的时间进行比较然后返回一个新的addflagpd的DataFrame
只比较分钟
"""


def changeDataFrame(serverpd: pd.DataFrame, addflagpd: pd.DataFrame) -> pd.DataFrame:
    addflagpd = addflagpd.copy()
    if FAULT_FLAG in addflagpd.columns.array:
        return pd.DataFrame()
    serverTimesList = serverpd[TIME_COLUMN_NAME].tolist()
    serverFlagsList = serverpd[FAULT_FLAG].tolist()
    addflagpdTimeList = addflagpd[TIME_COLUMN_NAME].tolist()
    addflagpdFlagList = []
    for itime in addflagpdTimeList:
        if itime not in serverTimesList:
            addflagpdFlagList.append(0)
            continue
        timepos = serverTimesList.index(itime)
        addflagpdFlagList.append(serverFlagsList[timepos])
    addflagpd[FAULT_FLAG] = addflagpdFlagList
    return addflagpd


def saveDirPdFromDict(spath, PDDict: Dict):
    for keypath, valuepd in PDDict.items():
        if len(valuepd) == 0:
            continue
        tpath = os.path.join(spath, keypath)
        filename = F[keypath]
        savepdfile(valuepd, tpath, filename)


if __name__ == "__main__":
    dirs = getDirs(R"DATA/2022-01-14新的测试数据")
    dirs = addFlagDir
    for idir in dirs:
        PD = getOneDirPd(dirpath=idir)
        savePD = {}
        savePD["server"] = PD["server"]
        savePD["process"] = changeDataFrame(PD["server"], PD["process"])
        savePD["l2"] = changeDataFrame(PD["server"], PD["l2"])
        savePD["network"] = changeDataFrame(PD["server"], PD["network"])
        savePD["ping"] = changeDataFrame(PD["server"], PD["ping"])
        savePD["topdown"] = changeDataFrame(PD["server"], PD["topdown"])
        saveDirPdFromDict(idir, savePD)
