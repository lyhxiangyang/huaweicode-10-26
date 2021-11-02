import json
import os
from typing import Dict, List

import pandas as pd

from Classifiers.ModelPred import select_and_pred
from utils.DataScripts import TranslateTimeToInt, getTime_AbnormalCore, TranslateTimeToStr
from utils.DataScripts import TranslateTimeToInt as c, getTime_AbnormalCore
from utils.DefineData import MODEL_TYPE, FAULT_FLAG, TIME_COLUMN_NAME
from utils.FileSaveRead import readFilename_Time_Core_pdDict, saveFilename_Time_Core_pdDict

"""
将每个核心上的数据都进行预测
"""
def predictFilename_Time_Core(ftcPD: Dict, modelpath: str):
    filename_time_corePd = {}
    for filename, time_core_pdDict in ftcPD.items():
        filename_time_corePd[filename] = {}
        for time, core_pdDict in time_core_pdDict.items():
            filename_time_corePd[filename][time] = {}
            for icore, tpd in core_pdDict.items():
                tpd: pd.DataFrame
                print("{}-{}-{}".format(filename, time, icore))
                for itype in MODEL_TYPE:
                    prelist = select_and_pred(tpd, model_type=itype, saved_model_path=modelpath)
                    tpd[itype + "_flag"] = prelist


abnormaliTime = [
    [c("2021-08-30 13:15:00"), c("2021-08-30 13:36:00")],
    [c("2021-08-30 13:55:00"), c("2021-08-30 14:16:00")],
    [c("2021-08-30 14:35:00"), c("2021-08-30 14:56:00")],
    [c("2021-08-30 15:15:00"), c("2021-08-30 15:36:00")],
    [c("2021-08-30 15:55:00"), c("2021-08-30 16:16:00")],
]
# 判断一个时间是否属于异常时间段内
def judgeTimeIsAbnormal(nowtime: str, abnormaltimes: List) -> bool:
    inowtime = c(nowtime)
    for i in abnormaltimes:
        if inowtime >= i[0] and inowtime <= i[1]:
            return True
    return False


if __name__ == "__main__":
    rmodelpath = "Classifiers/saved_model/tmp_load1_nosuffix"
    rpath = "tmp/tData-10-26/多机-E5-process-3KM"
    step6name = "6.filename-time-core-标准化-特征提取-未处理首尾"
    spath = "tmp/E5预测80数据"
    predictBegintime = "2021-08-30 13:14:00"
    predictEndtime = "2021-08-30 17:03:00"

    # 将未处理首尾的特征提取之后的数据进行读取
    tpath = os.path.join(rpath, step6name)
    filename_time_corePdDict = readFilename_Time_Core_pdDict(tpath,readfilename=["wrf_3km_e5-43_process-63", "wrf_3km_e5-43_process-64"] ,readtime=[0])
    # 进行预测
    predictFilename_Time_Core(filename_time_corePdDict, modelpath=rmodelpath)
    # 数据保存
    tpath = os.path.join(spath, "Data")
    saveFilename_Time_Core_pdDict(tpath, filename_time_corePdDict)

    # 进行解析时间和预测的关系
    tree_time_abnormalCoreDict, forest_time_abnormalCoreDict, adapt_time_abnormalCoreDict = getTime_AbnormalCore(filename_time_corePdDict)

    # 绘制关键的图 时间段
    predictBeginitime = TranslateTimeToInt(predictBegintime)
    predictEnditime = TranslateTimeToInt(predictEndtime)
    draw_time_flagDict = {}
    while (predictBeginitime <= predictEnditime):
        stime = TranslateTimeToStr(predictBeginitime)
        itime = predictBeginitime
        if stime not in draw_time_flagDict:
            draw_time_flagDict[stime] = {}
        # 真实标签 =====
        draw_time_flagDict[stime][FAULT_FLAG] = 0
        if (judgeTimeIsAbnormal(stime, abnormaliTime)):
            draw_time_flagDict[stime][FAULT_FLAG] = 30

        if stime not in tree_time_abnormalCoreDict:
            # print("{} 不在决策树中".format(stime))
            tree_time_abnormalCoreDict[stime] = []
        if stime not in forest_time_abnormalCoreDict:
            # print("{} 不在随机森林中".format(stime))
            forest_time_abnormalCoreDict[stime] = []
        if stime not in adapt_time_abnormalCoreDict:
            # print("{} 不在自适应增强中".format(stime))
            adapt_time_abnormalCoreDict[stime] = []
        # 决策树标签 =====
        draw_time_flagDict[stime][MODEL_TYPE[0] + "_flag"] = 0
        if len(tree_time_abnormalCoreDict[stime]) != 0:
            draw_time_flagDict[stime][MODEL_TYPE[0] + "_flag"] = 30
        # 随机森林标签 ====
        draw_time_flagDict[stime][MODEL_TYPE[1] + "_flag"] = 0
        if len(forest_time_abnormalCoreDict[stime]) != 0:
            draw_time_flagDict[stime][MODEL_TYPE[1] + "_flag"] = 30

        # 自适应增强标签 ====
        draw_time_flagDict[stime][MODEL_TYPE[2] + "_flag"] = 0
        if len(adapt_time_abnormalCoreDict[stime]) != 0:
            draw_time_flagDict[stime][MODEL_TYPE[2] + "_flag"] = 30
        predictBeginitime += 60

    tpd = pd.DataFrame(data=draw_time_flagDict).T
    tpd = tpd.reset_index()
    tpd = tpd.rename(columns={"index": "time"})
    tpd.to_csv(os.path.join(spath, "prelabels.csv"), index=False)

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



