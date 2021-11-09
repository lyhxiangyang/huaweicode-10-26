import os
from typing import List, Tuple, Union, Dict, Any

import pandas as pd

from utils.DataFrameOperation import subtractLastLineFromDataFrame
from utils.DefineData import TIME_COLUMN_NAME, TIME_INTERVAL, CPU_FEATURE, FAULT_FLAG, WINDOWS_SIZE
from utils.FileSaveRead import saveFaultyDict, saveFilename_Time_Core_pdDict, saveFilename_Time_Core_Faulty_pdDict, \
    saveDFListToFiles, saveCoreDFToFiles
from utils.DataScripts import TranslateTimeToInt, TranslateTimeListStrToStr, mergeTwoDF, splitDataFrameByTime, \
    SplitDFByCores, abstractFaultPDDict, processOneProcessFile

accumulationFeatures = ['user', 'system', 'iowait', 'read_count', 'write_count', 'read_bytes', 'write_bytes',
                        'read_chars', 'write_chars', 'voluntary', 'involuntary']
process_features = [
    # "time",
    # "pid",
    # "status",
    # "create_time",
    # "puids_real",
    # "puids_effective",
    # "puids_saved",
    # "pgids_real",
    # "pgids_effective",
    # "pgids_saved",
    "user",
    "system",
    # "children_user",
    # "children_system",
    "iowait",
    # "cpu_affinity",  # 依照这个来为数据进行分类
    "memory_percent",
    "rss",
    "vms",
    "shared",
    "text",
    "lib",
    "data",
    "dirty",
    "read_count",
    "write_count",
    "read_bytes",
    "write_bytes",
    "read_chars",
    "write_chars",
    "num_threads",
    "voluntary",
    "involuntary",
    "faultFlag",
]
datapath = [
    R"D:\HuaweiMachine\数据分类\TrainAndTest\测试数据\B\hpc环境\标注数据\hpc_wrf\wrf_3km_multi\hpcagent18\wrf_hpc_hpcagent18_process.csv",
]


# 将时间序列的秒这一项都变成秒
def changeTimeColumns_process(df: pd.DataFrame) -> pd.DataFrame:
    # 时间格式使用默认值
    tpd = df.loc[:, [TIME_COLUMN_NAME]].apply(lambda x: TranslateTimeListStrToStr(x.to_list(), timeformat="%Y/%m/%d %H:%M"), axis=0)
    df.loc[:, TIME_COLUMN_NAME] = tpd.loc[:, TIME_COLUMN_NAME]
    return df


if __name__ == "__main__":
    spath = "tmp/tData-11-09/测试数据/多机-HPC-process-3KM_123568"
    all_faulty_pd_dict = {}
    orginal_all_faulty_pd_dict = {}
    isSlideWin = True  # True代表这个step为1， False代表step为Win

    filename_time_core_pdDict = {}
    filename_time_core_faultDict = {}
    # 处理各个process文件
    for ipath in datapath:
        filename = os.path.basename(ipath)
        filename = os.path.splitext(filename)[0]

        # tmp/{}/0.合并server和process数据
        processpd = pd.read_csv(ipath)
        # 改变一个文件的时间， 因为server文件和process文件文件中的时间不对
        changeTimeColumns_process(processpd)

        # tmp/{filename}
        print("{} 处理".format(filename).center(40, "*"))
        onefile_Faulty_PD_Dict, time_core_pdDict, time_core_faultDict = processOneProcessFile(
            spath=os.path.join(spath, "0.所有文件处理过程", filename), filepd=processpd, accumulationFeatures=accumulationFeatures, process_features=process_features)
        all_faulty_pd_dict = mergeTwoDF(onefile_Faulty_PD_Dict, all_faulty_pd_dict)
        filename_time_core_pdDict[filename] = time_core_pdDict
        filename_time_core_faultDict[filename] = time_core_faultDict
    # 将所有的信息进行保存
    tallsavefaultypath = os.path.join(spath, "1.所有process错误码信息")
    saveFaultyDict(tallsavefaultypath, all_faulty_pd_dict)

    # 将filename-time-core进行保存
    sspath = os.path.join(spath, "2.filename-time-core")
    saveFilename_Time_Core_pdDict(sspath, filename_time_core_pdDict)

    # 将filename-time-core进行保存
    sspath = os.path.join(spath, "3.filename-time-core-faulty")
    saveFilename_Time_Core_Faulty_pdDict(sspath, filename_time_core_faultDict)
