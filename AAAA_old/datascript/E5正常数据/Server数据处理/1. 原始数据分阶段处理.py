import os

import pandas as pd

from utils.DataFrameOperation import mergeTwoDF
from utils.DataScripts import TranslateTimeListStrToStr, processOneServerFile
from utils.DefineData import TIME_COLUMN_NAME
from utils.FileSaveRead import saveFaultyDict, saveFilename_Time_pdDict, saveFilename_Time_Faulty_pdDict

accumulationFeatures = ['idle', 'iowait', 'interrupts', "usr_cpu", "kernel_cpu", 'ctx_switches', 'soft_interrupts', 'irq',
                  'softirq', 'steal', 'syscalls', 'handlesNum', 'pgpgin', 'pgpgout', 'fault', 'majflt', 'pgscank',
                  'pgsteal', "pgfree"]

server_feature = [
    # "time",
    "usr_cpu",
    "nice",
    "kernel_cpu",
    "idle",
    "iowait",
    "irq",
    "softirq",
    "steal",
    "guest",
    "guest_nice",
    "ctx_switches",
    "interrupts",
    "soft_interrupts",
    "syscalls",
    "freq",
    "load1",
    "load5",
    "load15",
    "total",
    "available",
    "percent",
    "mem_used",
    "free",
    "active",
    "inactive",
    "buffers",
    "cached",
    "handlesNum",
    "pgpgin",
    "pgpgout",
    "fault",
    "majflt",
    "pgscank",
    "pgsteal",
    "pgfree",
    # "faultFlag",
]
dataserverpath = [
    R"D:\HuaweiMachine\数据分类\wrf\多机\E5\3KM\正常数据\wrf_3km_mormal\wrf_e5_server_43.csv",
]


# 将时间序列的秒这一项都变成秒
def changeTimeColumns_process(df: pd.DataFrame) -> pd.DataFrame:
    # 时间格式使用默认值
    tpd = df.loc[:, [TIME_COLUMN_NAME]].apply(lambda x: TranslateTimeListStrToStr(x.to_list(), '%Y/%m/%d %H:%M'), axis=0)
    df.loc[:, TIME_COLUMN_NAME] = tpd.loc[:, TIME_COLUMN_NAME]
    return df


if __name__ == "__main__":
    spath = "tmp/tData-11-09/E5正常数据/server数据处理"
    all_faulty_pd_dict = {}
    orginal_all_faulty_pd_dict = {}
    isSlideWin = True  # True代表这个step为1， False代表step为Win

    filename_time_pdDict = {}
    filename_time_faultDict = {}
    # 处理各个process文件
    for ipath in dataserverpath:
        filename = os.path.basename(ipath)
        filename = os.path.splitext(filename)[0]

        # tmp/{}/0.合并server和process数据
        serverpd = pd.read_csv(ipath)
        # 改变一个文件的时间， 因为server文件和process文件文件中的时间不对
        changeTimeColumns_process(serverpd)

        # tmp/{filename}
        print("{} 处理".format(filename).center(40, "*"))

        onefile_Faulty_PD_Dict, time_pdDict, time_faultDict = processOneServerFile(
            spath=os.path.join(spath, "0.所有文件处理过程", filename), filepd=serverpd,
            accumulationFeatures=accumulationFeatures, server_features=server_feature)

        all_faulty_pd_dict = mergeTwoDF(onefile_Faulty_PD_Dict, all_faulty_pd_dict)
        filename_time_pdDict[filename] = time_pdDict
        filename_time_faultDict[filename] = time_faultDict
# =======
    tallsavefaultypath = os.path.join(spath, "1.所有process错误码信息")
    saveFaultyDict(tallsavefaultypath, all_faulty_pd_dict)

    # 将filename-time-core进行保存
    sspath = os.path.join(spath, "2.filename-time")
    saveFilename_Time_pdDict(sspath, filename_time_pdDict)

    # 将filename-time-core进行保存
    sspath = os.path.join(spath, "3.filename-time-faulty")
    saveFilename_Time_Faulty_pdDict(sspath, filename_time_faultDict)
