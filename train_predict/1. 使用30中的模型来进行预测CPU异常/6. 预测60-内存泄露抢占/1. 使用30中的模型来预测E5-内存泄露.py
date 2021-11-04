import json
import os
from typing import List, Any, Union

import pandas as pd

from Classifiers.ModelPred import predictFilename_Time_Core
from utils.DataFrameOperation import mergeDataFrames
from utils.DataScripts import TranslateTimeToInt as c, getTime_AbnormalCore, getResultFromTimequantum
from utils.FileSaveRead import readFilename_Time_Core_pdDict, saveFilename_Time_Core_pdDict




if __name__ == "__main__":
    # 要读取的文件路径
    rpath = "tmp/tData-10-26/多机-E5-process-3KM/6.filename-time-core-标准化-特征提取-未处理首尾"
    spath = "tmp/30模型预测总集合/6.1.预测60数据-E5"
    # 模型路径
    rmodelpath = "Classifiers/saved_model/tmp_load1_nosuffix"
    # E5全CPU抢占-10主要分布在 64、65、66里面
    rfilenames = ["wrf_3km_e5-43_process-70", "wrf_3km_e5-43_process-71", "wrf_3km_e5-43_process-72"]
    # 要预测所有的时间段
    rtimes = None
    # 要预测所有的核心
    rcores = None
    abnormaliTime = [
        ## 70
        [c("2021-08-31 02:37:00"), c("2021-08-31 02:56:00"), 61],
        [c("2021-08-31 03:17:00"), c("2021-08-31 03:36:00"), 62],
        ## 71
        [c("2021-08-31 03:57:00"), c("2021-08-31 04:16:00"), 63],
        [c("2021-08-31 034:37:00"), c("2021-08-31 034:56:00"), 64],
        [c("2021-08-31 05:17:00"), c("2021-08-31 05:36:00"), 65],
        ## 72
        [c("2021-08-31 05:57:00"), c("2021-08-31 06:16:00"), 91],
        [c("2021-08-31 06:37:00"), c("2021-08-31 06:56:00"), 92],

    ]
    # 为了防止时间过长，可以设置多个
    begin_endTime = [
        ("2021-08-31 02:17:00", "2021-08-31 07:15:00")
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

