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

""
def getDirs(dirpaths) -> List[str]:
    dirlists = []
    dirnamess = os.listdir(dirpaths)
    for idir in dirnamess:
        tpath = os.path.join(dirpaths, idir)
        if os.path.isdir(tpath):
            dirlists.append(tpath)
    return dirlists
def getFiles(dirpaths) -> List[str]:
    filelists = []
    dirnamess = os.listdir(dirpaths)
    for idir in dirnamess:
        tpath = os.path.join(dirpaths, idir)
        if os.path.isfile(tpath):
            filelists.append(tpath)
    return filelists
"""
得到server目录 process文件 topdown路径
"""
def getServerProcessTopdownPath(dirpath) -> Dict:
    fileDict = {}
    for ifile in getFiles(dirpath):
        filetype = os.path.splitext(os.path.split(ifile)[1])[0].split("_")[-1]
        if filetype == "server":
            fileDict["server"] = changeTimeFromOnepd(getfilepd(ifile))
        elif filetype == "process":
            fileDict["process"] = changeTimeFromOnepd(getfilepd(ifile))
        elif filetype == "topdown":
            fileDict["topdown"] = changeTimeFromOnepd(getfilepd(ifile))
    return fileDict


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
        R"DATA/测试数据/Openfoam/input1",
    ]
    for idirpaths in dirpaths:
        for idir in getDirs(idirpaths): # 得到数据样例
            for inode in getDirs(idir):
                # 得到server目录 process文件 topdown路径
                print(inode)
                fileDict = getServerProcessTopdownPath(inode)
                if len(fileDict) != 3:
                    continue
                savePD = {}
                savePD["server"] = fileDict["server"]
                savePD["process"] = changeDataFrame(fileDict["server"], fileDict["process"])
                savePD["topdown"] = changeDataFrame(fileDict["server"], fileDict["topdown"])
                saveDirPdFromDict(inode, savePD)
