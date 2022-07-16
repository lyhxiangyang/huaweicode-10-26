import os
from typing import List, Dict

import pandas as pd

from hpc.l3l2utils.DataFrameSaveRead import getfilepd, savepdfile
from hpc.l3l2utils.DataOperation import changeTimeFromOnepd, getsametimepd
from hpc.l3l2utils.DefineData import FAULT_FLAG, TIME_COLUMN_NAME

"""
这个文件用来根据server数据的时间对其他文件打上标签
"""

F = {
    "server": "hpc_server.csv",
    "process": "hpc_process.csv",
    "compute": "compute.csv",
    "nic": "nic.csv",
    "ping": "ping.csv",
    "topdown": "hpc_topdown.csv",
}


def getDirs(dirpaths) -> List[str]:
    dirnamess = os.listdir(dirpaths)
    dirpaths = [os.path.join(dirpaths, idir) for idir in dirnamess]
    return dirpaths


def getOneDirPd(dirpath: str):
    resDict = {}
    resDict["server"] = changeTimeFromOnepd(getfilepd(dirpath, F["server"]))
    resDict["process"] = changeTimeFromOnepd(getfilepd(dirpath, F["process"]))
    resDict["topdown"] = changeTimeFromOnepd(getfilepd(dirpath, F["topdown"]))
    resDict["compute"] = None
    resDict["nic"] = None
    resDict["ping"] = None
    return resDict


"""
根据server文件中的时间和addflag的时间进行比较然后返回一个新的addflagpd的DataFrame
只比较分钟
"""


def saveDirPdFromDict(spath, PDDict: Dict):
    for keypath, valuepd in PDDict.items():
        if valuepd is None:
            continue
        if len(valuepd) == 0:
            continue
        tpath = os.path.join(spath, keypath)
        filename = F[keypath]
        savepdfile(valuepd, tpath, filename)


"""
根据server文件中的时间和addflag的时间进行比较然后返回一个新的addflagpd的DataFrame
只比较分钟
这个是用来将addflagpd的标签根据serverpd来更改
"""

def changeDataFrame(serverpd: pd.DataFrame, addflagpd: pd.DataFrame) -> pd.DataFrame:
    addflagpd = addflagpd.copy()
    if FAULT_FLAG in addflagpd.columns.array:
        return addflagpd
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



"""
分析process数据，只取其中一部分
"""
def getrightProcesspd(processpd: pd.DataFrame):
    tpd = processpd[processpd["pname"] == "simpleFoam"]
    return tpd

if __name__ == "__main__":
    dirpaths = [
        R"csvfile_huawei/huawei_cachegrab90/intensity1",
        R"csvfile_huawei/huawei_cachegrab90/intensity2",
        R"csvfile_huawei/huawei_cachegrab90/intensity3",
        R"csvfile_huawei/huawei_memory_bandwidth50/intensity1",
        R"csvfile_huawei/huawei_memory_bandwidth50/intensity1_2",
        R"csvfile_huawei/huawei_memory_bandwidth50/intensity2",
        R"csvfile_huawei/huawei_memory_bandwidth50/intensity3",
        R"csvfile_huawei/allcpu10/intensity1",
        R"csvfile_huawei/allcpu10/intensity2",
        R"csvfile_huawei/allcpu10/intensity3",
        R"csvfile_huawei/multicpu30/intensity1",
        R"csvfile_huawei/multicpu30/intensity2",
        R"csvfile_huawei/multicpu30/intensity3",
        R"csvfile_huawei/singlecpu20/intensity1",
        R"csvfile_huawei/singlecpu20/intensity2",
        R"csvfile_huawei/singlecpu20/intensity3",
        R"csvfile_huawei/randomcpu80/intensity1",
        R"csvfile_huawei/randomcpu80/intensity2",
        R"csvfile_huawei/randomcpu80/intensity3",
    ]
    for idirpaths in dirpaths:
        dirs = getDirs(idirpaths)
        for idir in dirs:
            # 存在 server 就continue 不进行生成
            if os.path.exists(os.path.join(idir, "server")):
                continue
            # 有server目录就不进行下去
            print(idir)
            PD = getOneDirPd(dirpath=idir)
            savePD = {}
            # 获得正确的process文件
            savePD["process"] = getrightProcesspd(PD["process"])
            savePD["server"], _ = getsametimepd(PD["server"], savePD["process"])
            savePD["compute"] = None
            savePD["nic"] = None
            savePD["ping"] = None
            savePD["topdown"], _ = getsametimepd(PD["topdown"], savePD["process"])
            saveDirPdFromDict(idir, savePD)
