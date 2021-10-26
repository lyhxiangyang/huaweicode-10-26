import os

import pandas as pd

from utils.DataScripts import getDFmean
from utils.FileSaveRead import readFilename_Time_pdDict, readFilename_Time_Faulty_pdDict

if __name__ == "__main__":
    # 数据保存路径
    spath = "tmp/tData-10-26/多机-E5-process-3KM"
    # 所有正常数据的路径
    normalpath = "tmp/tData-10-26/多机-E5-server-3KM/1.所有process错误码信息/0.csv"
    # 需要标准化的数据路径
    file_timePath = "tmp/tData-10-26/多机-E5-server-3KM/2.filename-time"
    file_time_faultyPath: str = "tmp/tData-10-26/多机-E5-server-3KM/3.filename-time-faulty"
    # 需要标准化的特征
    standardfeatur = ["load1", "used"]
    standardvalue = 100

    # 获得所有正常情况下各个特征的平均值
    nomalpd = pd.read_csv(normalpath)
    normalmean = getDFmean(nomalpd, standardFeatures=standardfeatur)
    # 这个平均值我一定要保存下来
    print(normalmean)
    normalmean.to_csv(os.path.join(spath, "标准化平均值.csv"))

    # 读取所有的文件到字典中
    print("开始读取file_time信息".center(40, "*"))
    file_time_corePDDict = readFilename_Time_pdDict(file_timePath)
    print("结束读取file_time信息".center(40, "*"))
    print("开始读取file_time_faulty信息".center(40, "*"))
    file_time_core_faultyPDDict = readFilename_Time_Faulty_pdDict(file_time_faultyPath)
    print("结束读取file_time_faulty信息".center(40, "*"))

    # print("标准化开始处理".center(40, "*"))
    # file_time_corePDDict = standard_file_time_coreDict(file_time_corePDDict, standardFeature=standardfeatur,
    #                                                    meanvalue=normalmean, standardValue=standardvalue)
    # file_time_core_faultyPDDict = standard_file_time_core_faultyDict(file_time_core_faultyPDDict,
    #                                                                  standardFeature=standardfeatur,
    #                                                                  meanvalue=normalmean, standardValue=standardvalue)
    # print("标准化处理结束".center(40, "*"))
    #
    # # 将filename-time-core进行保存
    # sspath = os.path.join(spath, "4.filename-time-core-标准化")
    # saveFilename_Time_Core_pdDict(sspath, file_time_corePDDict)
    #
    # # 将filename-time-core进行保存
    # sspath = os.path.join(spath, "5.filename-time-core-faulty-标准化")
    # saveFilename_Time_Core_Faulty_pdDict(sspath, file_time_core_faultyPDDict)
