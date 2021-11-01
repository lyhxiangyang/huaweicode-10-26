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
    [c("2021-07-29 14:21:00"), c("2021-07-29 14:40:00")],
    [c("2021-07-29 14:49:00"), c("2021-07-29 15:10:00")],
    [c("2021-07-29 15:19:00"), c("2021-07-29 15:40:00")],
    [c("2021-07-29 15:49:00"), c("2021-07-29 16:10:00")],
    [c("2021-07-29 16:19:00"), c("2021-07-29 16:40:00")],
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
    predictBegintime = "2021-07-29 14:21:00"
    predictEndtime = "2021-07-29 16:48:00"
    predictBeginitime = TranslateTimeToInt(predictBegintime)
    predictEnditime = TranslateTimeToInt(predictEndtime)
    draw_time_flagDict = {}
    while (predictBeginitime <= predictEnditime):
        stime = TranslateTimeToStr(predictBeginitime)
        if stime not in draw_time_flagDict:
            draw_time_flagDict[stime] = {}
        # 真实标签 =====
        if (judgeTimeIsAbnormal(stime, abnormaliTime)):
            draw_time_flagDict[stime]["realflag"] = 30

        if stime not in tree_time_abnormalCoreDict:
            print("{} 不在决策树中".format(stime))
            tree_time_abnormalCoreDict[stime] = []
        if stime not in forest_time_abnormalCoreDict:
            print("{} 不在随机森林中".format(stime))
            forest_time_abnormalCoreDict[stime] = []
        if stime not in adapt_time_abnormalCoreDict:
            print("{} 不在自适应增强中".format(stime))
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

    tpd = pd.DataFrame(data=draw_time_flagDict).T
    tpd.to_csv(os.path.join(spath, "prelabels.csv"))


