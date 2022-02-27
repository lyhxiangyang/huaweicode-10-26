import os

import pandas as pd

from utils.DataScripts import TranslateTimeListStrToStr, mergeTwoDF, processOneProcessFile
from utils.DefineData import TIME_COLUMN_NAME
from utils.FileSaveRead import saveFaultyDict, saveFilename_Time_Core_pdDict, saveFilename_Time_Core_Faulty_pdDict

accumulationFeatures = ["usr_cpu", "kernel_cpu", 'iowait', 'read_count', 'write_count', 'read_bytes', 'write_bytes',
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
    "usr_cpu",
    "kernel_cpu",
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
    "D:/HuaweiMachine/数据分类/wrf/多机/E5/3KM/异常数据/wrf_3km_multi_43/wrf_3km_e5-43_process-61.csv",
    "D:/HuaweiMachine/数据分类/wrf/多机/E5/3KM/异常数据/wrf_3km_multi_43/wrf_3km_e5-43_process-62.csv",
    "D:/HuaweiMachine/数据分类/wrf/多机/E5/3KM/异常数据/wrf_3km_multi_43/wrf_3km_e5-43_process-63.csv",
    "D:/HuaweiMachine/数据分类/wrf/多机/E5/3KM/异常数据/wrf_3km_multi_43/wrf_3km_e5-43_process-64.csv",
    "D:/HuaweiMachine/数据分类/wrf/多机/E5/3KM/异常数据/wrf_3km_multi_43/wrf_3km_e5-43_process-65.csv",
    "D:/HuaweiMachine/数据分类/wrf/多机/E5/3KM/异常数据/wrf_3km_multi_43/wrf_3km_e5-43_process-66.csv",
    "D:/HuaweiMachine/数据分类/wrf/多机/E5/3KM/异常数据/wrf_3km_multi_43/wrf_3km_e5-43_process-67.csv",
    "D:/HuaweiMachine/数据分类/wrf/多机/E5/3KM/异常数据/wrf_3km_multi_43/wrf_3km_e5-43_process-68.csv",
    "D:/HuaweiMachine/数据分类/wrf/多机/E5/3KM/异常数据/wrf_3km_multi_43/wrf_3km_e5-43_process-69.csv",
    "D:/HuaweiMachine/数据分类/wrf/多机/E5/3KM/异常数据/wrf_3km_multi_43/wrf_3km_e5-43_process-70.csv",
    "D:/HuaweiMachine/数据分类/wrf/多机/E5/3KM/异常数据/wrf_3km_multi_43/wrf_3km_e5-43_process-71.csv",
    "D:/HuaweiMachine/数据分类/wrf/多机/E5/3KM/异常数据/wrf_3km_multi_43/wrf_3km_e5-43_process-72.csv",
    "D:/HuaweiMachine/数据分类/wrf/多机/E5/3KM/异常数据/wrf_3km_multi_43/wrf_3km_e5-43_process-73.csv",
    "D:/HuaweiMachine/数据分类/wrf/多机/E5/3KM/异常数据/wrf_3km_multi_43/wrf_3km_e5-43_process-74.csv",
    "D:/HuaweiMachine/数据分类/wrf/多机/E5/3KM/异常数据/wrf_3km_multi_43/wrf_3km_e5-43_process-75.csv",
    "D:/HuaweiMachine/数据分类/wrf/多机/E5/3KM/异常数据/wrf_3km_multi_43/wrf_3km_e5-43_process-76.csv",
    "D:/HuaweiMachine/数据分类/wrf/多机/E5/3KM/异常数据/wrf_3km_multi_43/wrf_3km_e5-43_process-77.csv",
    "D:/HuaweiMachine/数据分类/wrf/多机/E5/3KM/异常数据/wrf_3km_multi_43/wrf_3km_e5-43_process-78.csv",
    "D:/HuaweiMachine/数据分类/wrf/多机/E5/3KM/异常数据/wrf_3km_multi_43/wrf_3km_e5-43_process-79.csv",
    "D:/HuaweiMachine/数据分类/wrf/多机/E5/3KM/异常数据/wrf_3km_multi_43/wrf_3km_e5-43_process-80.csv",
    "D:/HuaweiMachine/数据分类/wrf/多机/E5/3KM/异常数据/wrf_3km_multi_43/wrf_3km_e5-43_process-81.csv",
]


# 将时间序列的秒这一项都变成秒
def changeTimeColumns_process(df: pd.DataFrame) -> pd.DataFrame:
    # 时间格式使用默认值
    tpd = df.loc[:, [TIME_COLUMN_NAME]].apply(lambda x: TranslateTimeListStrToStr(x.to_list()), axis=0)
    df.loc[:, TIME_COLUMN_NAME] = tpd.loc[:, TIME_COLUMN_NAME]
    return df


if __name__ == "__main__":
    spath = "tmp/tData-10-26/多机-E5-process-3KM"
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
