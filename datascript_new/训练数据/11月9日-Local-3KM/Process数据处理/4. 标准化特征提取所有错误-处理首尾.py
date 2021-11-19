import os

from utils.DataScripts import FeaExtra_file_time_core
from utils.FileSaveRead import readFilename_Time_Core_pdDict, saveFilename_Time_Core_pdDict, saveFaultyDict

if __name__ == "__main__":
    spath = "tmp/tData-11-09/训练数据/多机-Local-process-3KM"
    extractedFeaturee = ["cpu", "system", "user"]
    # 将所有的标准化数据读取
    file_time_core_standardPath = os.path.join(spath, "4.filename-time-core-标准化")
    print("读取filename-time-core数据中".center(40, "*"))
    file_time_core_standardDict = readFilename_Time_Core_pdDict(file_time_core_standardPath)
    # 进行特征提取
    print("特征提取中".center(40, "*"))
    file_time_core_standard_FeatureExtractionDict, allFault_PDDict = FeaExtra_file_time_core(file_time_core_standardDict, windowSize=3, windowRealSize=3, silidWindows=True,
                                                                                             extraFeature=extractedFeaturee)
    # 将特征提取之后的文件进行保存
    print("filename-time-core-标准化-特征提取开始".center(40, "*"))
    sspath = os.path.join(spath, "8.filename-time-core-标准化-特征提取-处理首尾")
    saveFilename_Time_Core_pdDict(sspath, file_time_core_standard_FeatureExtractionDict)
    print("filename-time-core-标准化-特征提取结束".center(40, "*"))

    # 将获得的所有特征提取之后的错误进行保存
    sspath = os.path.join(spath, "9.特征提取所有错误-处理首尾")
    saveFaultyDict(sspath, allFault_PDDict)
