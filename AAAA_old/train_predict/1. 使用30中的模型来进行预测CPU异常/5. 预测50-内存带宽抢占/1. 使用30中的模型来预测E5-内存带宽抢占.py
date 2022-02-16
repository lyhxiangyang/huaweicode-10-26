import json
import os

from hpc.Classifiers import predictFilename_Time_Core
from utils.DataFrameOperation import mergeDataFrames
from utils.DataScripts import TranslateTimeToInt as c, getTime_AbnormalCore, getResultFromTimequantum
from utils.FileSaveRead import readFilename_Time_Core_pdDict, saveFilename_Time_Core_pdDict

if __name__ == "__main__":
    # 要读取的文件路径
    rpath = "tmp/tData-10-26/多机-E5-process-3KM/6.filename-time-core-标准化-特征提取-未处理首尾"
    spath = "tmp/30模型预测总集合/5.1.预测50数据-E5"
    # 模型路径
    rmodelpath = "Classifiers/saved_model/tmp_load1_nosuffix"
    # E5全CPU抢占-10主要分布在 64、65、66里面
    rfilenames = ["wrf_3km_e5-43_process-61", "wrf_3km_e5-43_process-62"]
    # 要预测所有的时间段
    rtimes = None
    # 要预测所有的核心
    rcores = None
    # 三个个文件中异常分布的情况，因为并不都是10，11，12，13，14，15这五种异常
    abnormaliTime = [
        ## 61
        [c("2021-08-30 09:57:00"), c("2021-08-30 10:16:00"), 51],
        [c("2021-08-30 10:37:00"), c("2021-08-30 10:56:00"), 52],
        [c("2021-08-30 11:17:00"), c("2021-08-30 11:36:00"), 53],
        ## 62
        [c("2021-08-30 11:57:00"), c("2021-08-30 12:16:00"), 54],
        [c("2021-08-30 12:37:00"), c("2021-08-30 12:56:00"), 55],
    ]
    # 为了防止时间过长，可以设置多个
    begin_endTime = [
        ("2021-08-30 09:45:00", "2021-08-30 13:13:00")
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

