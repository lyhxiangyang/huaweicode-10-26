import os
from typing import List

import pandas as pd

from utils.DataFrameOperation import mergeDataFrames
from utils.DataScripts import splitDataFrameByTime
from utils.DefineData import TIME_COLUMN_NAME, FAULT_FLAG
from utils.FileSaveRead import saveDFListToFiles

datapath = [
    "D:/HuaweiMachine/数据分类/wrf/多机/红区/3KM/异常数据/wrf_3km_160_process.csv",
]


"""
传一个process的DF，然后将每一个时间点的user和system的和计算出来
"""


def getAllprocessCPUTime(processDF: pd.DataFrame, extractFeature: List[str]) -> pd.DataFrame:
    # 先每一行减去前一行, 根据pid进行处理
    processDF[extractFeature] = processDF.groupby("pid")[extractFeature].diff(periods=1)
    # 对每一个时间点进行求和
    user_systemDF = processDF.groupby(TIME_COLUMN_NAME)[extractFeature].sum()
    flagDF = processDF.groupby(TIME_COLUMN_NAME)[FAULT_FLAG].first()
    user_systemDF[FAULT_FLAG] = flagDF
    user_systemDF = user_systemDF.reset_index()
    return user_systemDF


def processAllprocessData(spath: str, datapath: List[str], extractFeature: List[str]) -> pd.DataFrame:
    if not os.path.exists(spath):
        os.makedirs(spath)
    # 进行文件的读取
    datapd = []
    for ipath in datapath:
        if not os.path.exists(ipath):
            print("{} 文件不存在".format(ipath))
            exit(1)
        print("处理文件-{}".format(ipath))
        tpd = pd.read_csv(ipath)
        # tpdlists = splitDataFrameByTime(tpd, time_interval=60, timeformat='%Y/%m/%d %H:%M')
        # datapd.extend(tpd)
        datapd.append(tpd)

    # 进行文件的分析
    cpudatapd = []
    for ipd in datapd:
        tpd = getAllprocessCPUTime(ipd, extractFeature)
        cpudatapd.append(tpd)
    saveDFListToFiles(os.path.join(spath, "2. 处理积累数据"), cpudatapd)
    # 进行文件的合并
    mergecpupd, _ = mergeDataFrames(cpudatapd)
    # 使用新特征cpu
    mergecpupd["cpu"] = mergecpupd["user"] + mergecpupd["system"]
    mergecpupd: pd.DataFrame
    mergecpupd.to_csv(os.path.join(spath, "mergedcpu.csv"), index=False)
    return mergecpupd


if __name__ == "__main__":
    spath = "tmp/tData-10-26/多机-Local-process-3KM/10.提取进程文件中的CPU数据"
    extractFeature = ["user", "system"]
    processAllprocessData(spath, datapath, extractFeature)
