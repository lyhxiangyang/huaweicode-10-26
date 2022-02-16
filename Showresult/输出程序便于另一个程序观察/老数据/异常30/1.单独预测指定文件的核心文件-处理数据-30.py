
"""
单独指定datascript中步骤3中生成的每个文件每个核心的文件使用现有模型进行预测，预测结果存储在
"""
import os.path

import pandas as pd

from AAAA_old.TrainToTest import testThree
# 将一个DataFrame的FAULT_FLAG重值为ff
from utils.DataFrameOperation import mergeDataFrames
from utils.DefineData import FAULT_FLAG


def setPDfaultFlag(df: pd.DataFrame) -> pd.DataFrame:
    realflag = list(df[FAULT_FLAG])
    if FAULT_FLAG in df.columns.array:
        df = df.drop(FAULT_FLAG, axis=1)
    dealflag = [(i // 10) * 10 for i in realflag]
    ffdict = {FAULT_FLAG: dealflag}
    tpd = pd.DataFrame(data=ffdict)
    tpd = pd.concat([df, tpd], axis=1)
    return tpd

if __name__ == "__main__":
    spath = "tmp/allpreinformation"
    # 使用模型路径
    usemodelpath = "Classifiers/saved_model/tmp_load1_nosuffix"
    # 要预测的文件
    prefiles = [
        "tmp/tData-10-26/多机-Local-process-3KM/8.filename-time-core-标准化-特征提取-处理首尾/wrf_3km_160_process/1/0.csv",
    ]

    # 要预测文件的路径
    pdlist = []
    for ipath in prefiles:
        if not os.path.exists(ipath):
            print("文件不存在")
            exit(1)
        prepd = pd.read_csv(ipath)
        prepd = setPDfaultFlag(prepd)
        pdlist.append(prepd)
    prepd, err = mergeDataFrames(pdlist)
    testThree(prepd, spath=spath, modelpath=usemodelpath)

