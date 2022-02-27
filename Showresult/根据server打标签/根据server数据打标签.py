import os
from typing import List, Dict

import pandas as pd

from hpc.l3l2utils.DataFrameSaveRead import getfilepd, savepdfile
from hpc.l3l2utils import changeTimeFromOnepd
from hpc.l3l2utils import TIME_COLUMN_NAME, FAULT_FLAG

"""
这个文件用来根据server数据的时间对其他文件打上标签
"""

F = {
    "server": "metric_server.csv",
    "process": "hpc_process.csv",
    "compute": "compute.csv",
    "network": "nic.csv",
    "ping": "ping.csv",
    "topdown": "topdown.csv",
}
addFlagDir = [
    R"C:\Users\lWX1084330\Desktop\json输入输出格式\test_all\test\grapes_test1p_multi_l3\centos11"
]


def getDirs(dirpaths) -> List[str]:
    dirnamess = os.listdir(dirpaths)
    dirlists = [os.path.join(dirpaths, idir, "centos11") for idir in dirnamess]
    return dirlists


def getOneDirPd(dirpath: str):
    resDict = {}
    resDict["server"] = changeTimeFromOnepd(getfilepd(dirpath, F["server"]))
    resDict["process"] = changeTimeFromOnepd(getfilepd(dirpath, F["process"]))
    resDict["compute"] = changeTimeFromOnepd(getfilepd(dirpath, F["compute"]))
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
        tpath = os.path.join(spath, keypath)
        filename = F[keypath]
        savepdfile(valuepd, tpath, filename)


if __name__ == "__main__":
    # dirs = getDirs(R"C:\Users\lWX1084330\Desktop\json输入输出格式\test_all\test")
    dirs = addFlagDir
    for idir in dirs:
        PD = getOneDirPd(dirpath=idir)
        savePD = {}
        savePD["server"] = PD["server"]
        savePD["process"] = changeDataFrame(PD["server"], PD["process"])
        savePD["compute"] = changeDataFrame(PD["server"], PD["compute"])
        savePD["network"] = changeDataFrame(PD["server"], PD["network"])
        savePD["ping"] = changeDataFrame(PD["server"], PD["ping"])
        savePD["topdown"] = changeDataFrame(PD["server"], PD["topdown"])
        saveDirPdFromDict(idir, savePD)



