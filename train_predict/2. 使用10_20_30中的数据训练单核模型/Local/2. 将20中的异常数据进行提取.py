import os

from utils.DataScripts import FeaExtra_file_time_core
from utils.FileSaveRead import readFilename_Time_Core_pdDict, saveFilename_Time_Core_pdDict, saveFaultyDict

if __name__ == "__main__":
    rstandardpath = "tmp/tData-10-26/多机-Local-process-3KM/4.filename-time-core-标准化"
    # 一切过程数据都存储在spath路径下
    spath = "tmp/使用10_20_30中的数据训练单核模型/Local"
    # 要读取的filename
    rfilename = ["wrf_3km_160_process"]
    rtime = [1]
    rcores = [0]
    extractedFeatures = ["cpu"]

    # 读取数据
    print("读取数据中".center(40, "*"))
    file_time_core_standardDict = readFilename_Time_Core_pdDict(rstandardpath,readfilename=rfilename,readtime=rtime, readcore=rcores)
    print("读取数据结束".center(40, "*"))

    # 特征提取 - 处理数据
    print("特征提取中".center(40, "*"))
    file_time_core_standard_FeatureExtractionDict, allFault_PDDict = FeaExtra_file_time_core(file_time_core_standardDict, windowSize=3, windowRealSize=3, silidWindows=True,
                                                                                             extraFeature=extractedFeatures)
    print("特征提取结束".center(40, "*"))
    # 将特征提取之后的文件进行保存
    print("数据保存中".center(40, "*"))
    sspath = os.path.join(spath, "3.错误码20-标准化-特征提取-处理首尾")
    saveFilename_Time_Core_pdDict(sspath, file_time_core_standard_FeatureExtractionDict)
    print("数据保存结束".center(40, "*"))

    # 将获得的所有特征提取之后的错误进行保存
    sspath = os.path.join(spath, "4.错误码20-特征提取所有错误-处理首尾")
    saveFaultyDict(sspath, allFault_PDDict)





