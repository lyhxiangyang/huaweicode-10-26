# 需要标准化文件所在的路径
import os.path

import pandas as pd

# 需要标准化的错误码
from utils.DataScripts import getDFmean, standard_file_time_coreDict, standard_file_time_core_faultyDict
from utils.FileSaveRead import readFilename_Time_Core_pdDict, readFilename_Time_Core_Faulty_pdDict, \
    saveFilename_Time_Core_pdDict, saveFilename_Time_Core_Faulty_pdDict

standardized_normalflag = 0
standardized_abnormalflag = [15]
# # 所有需要标准化的特征 这个变量没有被使用 仅供参考
# allFeature = [
#     # "time",
#     # "pid",
#     # "status",
#     # "create_time",
#     # "puids_real",
#     # "puids_effective",
#     # "puids_saved",
#     # "pgids_real",
#     # "pgids_effective",
#     # "pgids_saved",
#     "usr_cpu",
#     "kernel_cpu",
#     # "children_user",
#     # "children_system",
#     "iowait",
#     # "cpu_affinity",  # 依照这个来为数据进行分类
#     "memory_percent",
#     "rss",
#     "vms",
#     "shared",
#     "text",
#     "lib",
#     "data",
#     "dirty",
#     "read_count",
#     "write_count",
#     "read_bytes",
#     "write_bytes",
#     "read_chars",
#     "write_chars",
#     "num_threads",
#     "voluntary",
#     "involuntary",
#     # "faultFlag",
# ]





if __name__ == "__main__":
    # 所有正常数据的路径
    spath = "tmp/tData-11-09/E5正常数据/process数据处理"

    # 使用自己的正常数据
    # normalreadpath = os.path.join(spath, R"1.所有process错误码信息\0.csv")
    # 使用正常数据中的平均值
    normalreadpath = R"D:\HuaweiMachine\huaweicode-10-26\tmp\tData-11-09\E5正常数据\process数据处理\1.所有process错误码信息\0.csv"
    normalpath = os.path.join(normalreadpath)
    # 需要标准化的数据路径
    file_time_corePath = os.path.join(spath, "2.filename-time-core")
    file_time_core_faultyPath = os.path.join(spath, "3.filename-time-core-faulty")
    # 需要标准化的特征
    standardfeatur = ["usr_cpu", "kernel_cpu", "cpu"]
    standardvalue = 60

    # 获得所有正常情况下各个特征的平均值
    nomalpd = pd.read_csv(normalpath)
    normalmean = getDFmean(nomalpd, standardFeatures=standardfeatur)
    # 这个平均值我一定要保存下来
    print(normalmean)
    normalmean.to_csv(os.path.join(spath, "标准化平均值.csv"))

    # 读取所有的文件到字典中
    print("开始读取file_time_core信息".center(40, "*"))
    file_time_corePDDict = readFilename_Time_Core_pdDict(file_time_corePath)
    print("结束读取file_time_core信息".center(40, "*"))
    print("开始读取file_time_core_faulty信息".center(40, "*"))
    file_time_core_faultyPDDict = readFilename_Time_Core_Faulty_pdDict(file_time_core_faultyPath)
    print("结束读取file_time_core_faulty信息".center(40, "*"))

    print("标准化开始处理".center(40, "*"))
    file_time_corePDDict = standard_file_time_coreDict(file_time_corePDDict, standardFeature=standardfeatur,
                                                       meanvalue=normalmean, standardValue=standardvalue)
    file_time_core_faultyPDDict = standard_file_time_core_faultyDict(file_time_core_faultyPDDict,
                                                                     standardFeature=standardfeatur,
                                                                     meanvalue=normalmean, standardValue=standardvalue)
    print("标准化处理结束".center(40, "*"))

    # 将filename-time-core进行保存
    sspath = os.path.join(spath, "4.filename-time-core-标准化")
    saveFilename_Time_Core_pdDict(sspath, file_time_corePDDict)

    # 将filename-time-core进行保存
    sspath = os.path.join(spath, "5.filename-time-core-faulty-标准化")
    saveFilename_Time_Core_Faulty_pdDict(sspath, file_time_core_faultyPDDict)
