import os

from utils.DataScripts import FeaExtra_file_time_core
from utils.FileSaveRead import readFilename_Time_Core_pdDict, saveFilename_Time_Core_pdDict, saveFaultyDict

if __name__ == "__main__":
    rstandardpath = "tmp/tData-10-26/多机-E5-process-3KM/4.filename-time-core-标准化"
    # 一切过程数据都存储在spath路径下
    spath = "tmp/使用10_20_30中的数据训练单核模型/E5"
    # 要读取的filename
    rfilename = ["wrf_3km_e5-43_process-64", "wrf_3km_e5-43_process-65", "wrf_3km_e5-43_process-66"]
    rtime = None
    rcores = None
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
    sspath = os.path.join(spath, "1.错误码10-标准化-特征提取-处理首尾")
    saveFilename_Time_Core_pdDict(sspath, file_time_core_standard_FeatureExtractionDict)
    print("数据保存结束".center(40, "*"))

    # 将获得的所有特征提取之后的错误进行保存
    sspath = os.path.join(spath, "2.错误码10-特征提取所有错误-处理首尾")
    saveFaultyDict(sspath, allFault_PDDict)





