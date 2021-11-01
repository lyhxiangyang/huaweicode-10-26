import os
from typing import Dict

import pandas as pd

from Classifiers.ModelPred import select_and_pred
from utils.DefineData import MODEL_TYPE, FAULT_FLAG, TIME_COLUMN_NAME
from utils.FileSaveRead import readFilename_Time_Core_pdDict, saveFilename_Time_Core_pdDict

"""
将每个核心上的数据
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


if __name__ == "__main__":
    rmodelpath = "Classifiers/saved_model/tmp_load1_nosuffix"
    rpath = "tmp/tData-10-26/多机-Local-process-3KM"
    step6name = "6.filename-time-core-标准化-特征提取-未处理首尾"
    spath = "tmp/Local预测80数据"

    # 将未处理首尾的特征提取之后的数据进行读取
    tpath = os.path.join(rpath, step6name)
    filename_time_corePdDict = readFilename_Time_Core_pdDict(tpath, readtime=[7])
    # 进行预测
    predictFilename_Time_Core(filename_time_corePdDict, modelpath=rmodelpath)
    # 数据保存
    saveFilename_Time_Core_pdDict(spath, filename_time_corePdDict)




