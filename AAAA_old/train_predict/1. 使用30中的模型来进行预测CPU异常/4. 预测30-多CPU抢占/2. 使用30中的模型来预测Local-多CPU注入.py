import json
import os

from hpc.Classifiers import predictFilename_Time_Core
from utils.DataFrameOperation import mergeDataFrames
from utils.DataScripts import TranslateTimeToInt as c, getTime_AbnormalCore, getResultFromTimequantum
from utils.FileSaveRead import readFilename_Time_Core_pdDict, saveFilename_Time_Core_pdDict

if __name__ == "__main__":
    # 要读取的文件路径
    rpath = "tmp/tData-10-26/多机-Local-process-3KM/6.filename-time-core-标准化-特征提取-未处理首尾"
    spath = "tmp/30模型预测总集合/4.2.预测30数据-Local"
    # 模型路径
    rmodelpath = "Classifiers/saved_model/tmp_load1_nosuffix"
    # Local全CPU抢占只有一个文件，默认即可
    rfilenames = None
    # 要预测所有的时间段 0, 1
    rtimes = [1]
    # 要预测所有的核心
    rcores = None
    # 三个个文件中异常分布的情况，因为并不都是10，11，12，13，14，15这五种异常
    abnormaliTime = [
        # 1时间段
        [c("2021-06-29 21:37:00"), c("2021-06-29 21:46:00"), 15],
        [c("2021-06-29 22:07:00"), c("2021-06-29 22:16:00"), 21],
        [c("2021-06-29 22:37:00"), c("2021-06-29 22:46:00"), 22],
        [c("2021-06-29 23:07:00"), c("2021-06-29 23:16:00"), 23],
        [c("2021-06-29 23:37:00"), c("2021-06-29 23:46:00"), 24],
        [c("2021-06-30 00:07:00"), c("2021-06-30 00:16:00"), 25],
        [c("2021-06-30 00:37:00"), c("2021-06-30 00:46:00"), 31],
        [c("2021-06-30 01:07:00"), c("2021-06-30 01:16:00"), 32],
        [c("2021-06-30 01:37:00"), c("2021-06-30 01:46:00"), 33],
        [c("2021-06-30 02:07:00"), c("2021-06-30 02:16:00"), 34],
        [c("2021-06-30 02:37:00"), c("2021-06-30 02:46:00"), 35],
    ]
    # 为了防止时间过长，可以设置多个
    begin_endTime = [
        ("2021-06-29 21:36:00", "2021-06-30 03:06:00")
    ]

    #=====================================进行标准流程
    # 将未处理首尾的特征提取之后的数据进行读取
    filename_time_corePdDict = readFilename_Time_Core_pdDict(rpath, readfilename=rfilenames, readtime=rtimes, readcore=rcores)
    # 进行预测, 数据会保存在filename_time_corePdDict字典里面
    predictFilename_Time_Core(filename_time_corePdDict, modelpath=rmodelpath)
    # 数据保存
    tpath = os.path.join(spath, "Data")
    saveFilename_Time_Core_pdDict(tpath, filename_time_corePdDict)

    # 进行解析时间和预测的关系, 将每个时间点的预测为异常的核心都存储为
    tree_time_abnormalCoreDict, forest_time_abnormalCoreDict, adapt_time_abnormalCoreDict = getTime_AbnormalCore(filename_time_corePdDict)

    # 对每个开始和结束时间进行循环
    time_resultpdlist = []
    for itime in begin_endTime:
        beingtime = itime[0]
        endtime = itime[1]
        time_resultPD = getResultFromTimequantum(
            predictBegintime=beingtime,
            predictEndtime=endtime,
            abnormaliTime=abnormaliTime,
            tree_time_abnormalCoreDict=tree_time_abnormalCoreDict,
            forest_time_abnormalCoreDict=forest_time_abnormalCoreDict,
            adapt_time_abnormalCoreDict=adapt_time_abnormalCoreDict
        )
        time_resultpdlist.append(time_resultPD)

    allresultpd, _ = mergeDataFrames(time_resultpdlist)

    # 将数据进行保存
    tfilepath = os.path.join(spath, "prelabels.csv")
    allresultpd.to_csv(tfilepath, index=False)
    # 将time-cores进行保存
    treejson = json.dumps(tree_time_abnormalCoreDict)
    forestjson = json.dumps(forest_time_abnormalCoreDict)
    adaptjson = json.dumps(adapt_time_abnormalCoreDict)
    with open(os.path.join(spath, "tree_time_cores.json"), "w") as f:
        f.write(str(treejson))
    with open(os.path.join(spath, "forest_time_cores.json"), "w") as f:
        f.write(str(forestjson))
    with open(os.path.join(spath, "adapt_time_cores.json"), "w") as f:
        f.write(str(adaptjson))

