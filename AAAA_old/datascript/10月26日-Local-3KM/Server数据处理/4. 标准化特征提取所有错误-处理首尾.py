import os

from utils.DataScripts import FeaExtra_file_time
from utils.FileSaveRead import readFilename_Time_pdDict, saveFilename_Time_pdDict, saveFaultyDict

if __name__ == "__main__":
    spath = "tmp/tData-10-26/多机-Local-server-3KM"
    # extractedFeaturee = ["load1", "mem_used"]
    extractedFeaturee = [
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
    # 将所有标准化数据进行读取
    file_time_standardPath = os.path.join(spath, "4.filename-time-标准化")
    print("读取filename-time-core数据中".center(40, "*"))
    file_time_core_standardDict = readFilename_Time_pdDict(file_time_standardPath)

    # 进行特征提取
    print("特征提取中".center(40, "*"))
    file_time_standard_FeatureExtractionDict, allFault_PDDict = FeaExtra_file_time(file_time_core_standardDict, windowSize=3, windowRealSize=3, silidWindows=True, extraFeature=extractedFeaturee)

    # 将特征提取之后的文件进行保存
    print("filename-time-标准化-特征提取开始".center(40, "*"))
    sspath = os.path.join(spath, "8.filename-time-标准化-特征提取-处理首尾")
    saveFilename_Time_pdDict(sspath, file_time_standard_FeatureExtractionDict)
    print("filename-time-标准化-特征提取结束".center(40, "*"))

    # 将获得的所有特征提取之后的错误进行保存
    sspath = os.path.join(spath, "9.特征提取所有错误-处理首尾")
    saveFaultyDict(sspath, allFault_PDDict)



