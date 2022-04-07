import os.path
from typing import Dict

import joblib

from hpc.l3l2utils.DefineData import MODEL_TYPE


def preproccessConfigfile(inputDict: Dict) -> Dict:
    # 对阈值重新进行设置，必须为决策树
    def changeMemLeakThread(modelpath: str, threads):
        f = open(modelpath, 'rb')
        model = joblib.load(f)
        f.close()
        model.tree_.threshold[0] = threads
        joblib.dump(model, modelpath)

    # 1. 预处理第一步对内存泄露的模型进行重新设置
    modelpath=os.path.join(inputDict["servermemory_modelpath"], MODEL_TYPE[inputDict["servermemory_modeltype"]] + ".pkl")
    modelthread=inputDict["memleakpermin"]
    changeMemLeakThread(modelpath, modelthread)

