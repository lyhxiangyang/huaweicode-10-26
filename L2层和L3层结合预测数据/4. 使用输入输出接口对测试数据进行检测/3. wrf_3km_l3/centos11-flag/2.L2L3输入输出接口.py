import os
import sys
import time
from typing import Dict

from hpc.l3l2utils.L2L3Main import detectionFromInputDict
from hpc.l3l2utils.ParsingJson import readJsonToDict

def JoinWorkingDirPathFromConfig(workpath: str, configJsonDict: Dict) -> Dict:
    # predictdirjsonpath
    keynames = [
        "predictdirjsonpath",
        "spath",
        "processcpu_modelpath",
        "servermemory_modelpath",
        "serverbandwidth_modelpath",
        "power_machine_modelpath",
        "power_cabinet_modelpath",
        "temperature_modelpath",
        "network_pfcpath",
        "network_tx_hangpath",
        "resultsavepath",
    ]
    for ikeynames in keynames:
        if ikeynames in configJsonDict and configJsonDict[ikeynames] is not None:
            configJsonDict[ikeynames] = os.path.join(workpath, configJsonDict[ikeynames])

if __name__ == "__main__":
    startTime = time.perf_counter()
    # configfilepath = R"L2层和L3层结合预测数据/4. 使用输入输出接口对测试数据进行检测/2.检测每个测试文件中的数据/2.config.json"
    configfilepath = os.path.join(sys.path[0], "2.config.json")
    configJsonDict = readJsonToDict(*(os.path.split(configfilepath)))
    # 将这个configJsonDict中的路径都加上工作路径

    outputDict = detectionFromInputDict(configJsonDict)
    endTime = time.perf_counter()
    print('Running time: %s Seconds' % (endTime - startTime))
