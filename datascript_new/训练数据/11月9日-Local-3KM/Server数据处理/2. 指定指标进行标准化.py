import os

import pandas as pd

from utils.DataScripts import getDFmean, standard_file_time_Dict, standard_file_time_faultyDict
from utils.FileSaveRead import readFilename_Time_pdDict, readFilename_Time_Faulty_pdDict, saveFilename_Time_pdDict, \
    saveFilename_Time_Faulty_pdDict

if __name__ == "__main__":
    # 数据保存路径
    spath = "tmp/tData-11-09/训练数据/多机-Local-server-3KM"

    # 使用自己的正常数据
    # normalreadpath = os.path.join(spath, R"1.所有process错误码信息\0.csv")
    # 所有正常数据的路径
    normalreadpath = R"D:\HuaweiMachine\huaweicode-10-26\tmp\tData-11-09\Local正常数据\server数据处理\1.所有process错误码信息\0.csv"
    normalpath = os.path.join(normalreadpath)

    # 需要标准化的数据路径
    file_timePath = os.path.join(spath, "2.filename-time")
    file_time_faultyPath = os.path.join(spath, "3.filename-time-faulty")
    # 需要标准化的特征
    # 只标准化这几个特征值
    # standardfeatur = ["load1", "used"]
    # 将所有的特征值都进行标准化
    standardfeatur = [
        # "time",
        "user",
        "nice",
        "system",
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
        "used",
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
    standardvalue = 100

    # 获得所有正常情况下各个特征的平均值
    nomalpd = pd.read_csv(normalpath)
    normalmean = getDFmean(nomalpd, standardFeatures=standardfeatur)
    # 这个平均值我一定要保存下来
    print(normalmean)
    normalmean.to_csv(os.path.join(spath, "标准化平均值.csv"))

    # 读取所有的文件到字典中
    print("开始读取file_time信息".center(40, "*"))
    file_time_PDDict = readFilename_Time_pdDict(file_timePath)
    print("结束读取file_time信息".center(40, "*"))
    print("开始读取file_time_faulty信息".center(40, "*"))
    file_time_faultyPDDict = readFilename_Time_Faulty_pdDict(file_time_faultyPath)
    print("结束读取file_time_faulty信息".center(40, "*"))

    print("标准化开始处理".center(40, "*"))
    file_time_PDDict = standard_file_time_Dict(file_time_PDDict, standardFeature=standardfeatur,
                                                       meanvalue=normalmean, standardValue=standardvalue)
    file_time_faultyPDDict = standard_file_time_faultyDict(file_time_faultyPDDict,
                                                                     standardFeature=standardfeatur,
                                                                     meanvalue=normalmean, standardValue=standardvalue)
    print("标准化处理结束".center(40, "*"))

    # 将filename-time-core进行保存
    sspath = os.path.join(spath, "4.filename-time-标准化")
    saveFilename_Time_pdDict(sspath, file_time_PDDict)

    # 将filename-time-core进行保存
    sspath = os.path.join(spath, "5.filename-time-faulty-标准化")
    saveFilename_Time_Faulty_pdDict(sspath, file_time_faultyPDDict)
