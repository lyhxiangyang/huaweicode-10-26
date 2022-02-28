import os
from pathlib import Path
from typing import List, Dict

import pandas as pd

from hpc.l3l2utils.DataFrameSaveRead import getfilepd, savepdfile
from hpc.l3l2utils.DataOperation import changeTimeFromOnepd
from hpc.l3l2utils.DefineData import FAULT_FLAG, TIME_COLUMN_NAME

"""
这个文件用来根据server数据的时间对其他文件打上标签
"""

F = {
    "server": "metric_server.csv",
    "process": "hpc_process.csv",
    "compute": "compute.csv",
    "nic": "nic.csv",
    "ping": "ping.csv",
    "topdown": "topdown.csv",
}
addFlagDir = [
    R"DATA/2022-01-14新的测试数据/25.未知异常错误找原因/centos11",
    R"DATA/2022-01-14新的测试数据/25.未知异常错误找原因/centos16",
    R"DATA/2022-01-14新的测试数据/25.未知异常错误找原因/centos21",
    R"DATA/2022-01-14新的测试数据/25.未知异常错误找原因/centos26",
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
    resDict["compute"] = changeTimeFromOnepd(getfilepd(dirpath, F["compute"]))
    resDict["nic"] = changeTimeFromOnepd(getfilepd(dirpath, F["nic"]))
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

def getDigitalListDir():
    pass
# changedigitrange = list(range(28,29))
if __name__ == "__main__":
    dirpaths = [
        R"DATA/测试数据/WRF/1KM",
        R"DATA/测试数据/WRF/3KM",
        R"DATA/测试数据/WRF/9KM",
        R"DATA/测试数据/Grapes/test1",
        R"DATA/测试数据/Grapes/国家超算",
    ]
    for idirpaths in dirpaths:
        dirs = getDirs(idirpaths)
        for idir in dirs:
            iidir = Path(idir).parent.name
            # dirnumber = int(iidir.split(".")[0])
            # if not dirnumber in changedigitrange:
            #     continue
            print(idir)
            PD = getOneDirPd(dirpath=idir)
            savePD = {}
            savePD["server"] = PD["server"]
            savePD["process"] = changeDataFrame(PD["server"], PD["process"])
            savePD["compute"] = changeDataFrame(PD["server"], PD["compute"])
            savePD["nic"] = changeDataFrame(PD["server"], PD["nic"])
            savePD["ping"] = changeDataFrame(PD["server"], PD["ping"])
            savePD["topdown"] = changeDataFrame(PD["server"], PD["topdown"])
            saveDirPdFromDict(idir, savePD)
