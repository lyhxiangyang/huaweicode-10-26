import os

from utils.DataScripts import FeaExtra_file_time_core
from utils.FileSaveRead import readFilename_Time_Core_pdDict, saveFilename_Time_Core_pdDict, saveFaultyDict

if __name__ == "__main__":
    filename = 'readcores未处理首尾.txt'
    spath = "tmp/预测30处理数据/多机-Local-process-3KM"
    extractedFeaturee = ["cpu", "kernel_cpu", "usr_cpu"]

    if not os.path.exists(spath):
        os.makedirs(spath)
    # 将所有的标准化数据读取
    file_time_core_standardPath = "tmp/tData-10-26/多机-Local-process-3KM/4.filename-time-core-标准化"
    print("读取filename-time-core数据中".center(40, "*"))
    readcores = list(range(0, 7))
    file_time_core_standardDict = readFilename_Time_Core_pdDict(file_time_core_standardPath, readcore=readcores)
    # 输出核心的信息 并且将其输出到文件中 ===============
    print("读取核心数：{}".format(readcores))
    filepath = os.path.join(spath, filename)
    with open(filepath, 'w') as file_object:
        file_object.write(str(readcores))
    # ============================================

    # 进行特征提取
    print("特征提取中".center(40, "*"))
    file_time_core_standard_FeatureExtractionDict, allFault_PDDict = FeaExtra_file_time_core(file_time_core_standardDict, windowSize=3, windowRealSize=1, silidWindows=True,
                                                                                             extraFeature=extractedFeaturee)
    # 将特征提取之后的文件进行保存
    print("filename-time-core-标准化-特征提取开始".center(40, "*"))
    sspath = os.path.join(spath, "1.filename-time-core-标准化-特征提取-未处理首尾")
    saveFilename_Time_Core_pdDict(sspath, file_time_core_standard_FeatureExtractionDict)
    print("filename-time-core-标准化-特征提取结束".center(40, "*"))

    # 将获得的所有特征提取之后的错误进行保存
    sspath = os.path.join(spath, "2.特征提取所有错误-未处理首尾")
    saveFaultyDict(sspath, allFault_PDDict)
