import os
import sys
import time

from hpc.l3l2utils.L2L3Main import detectionFromInputDict
from hpc.l3l2utils.ParsingJson import readJsonToDict
from hpc.l3l2utils.PreProcessConfig import preproccessConfigfile

if __name__ == "__main__":
    startTime = time.perf_counter()
    configfilepath = os.path.join(sys.path[0], "config.json")
    configJsonDict = readJsonToDict(*(os.path.split(configfilepath)))
    # 对config文件中的数据进行预处理
    preproccessConfigfile(configJsonDict)
    configJsonDict["predictdirjsonpath"] = configJsonDict["predictdirjsonpath"]
    configJsonDict["spath"] = configJsonDict["spath"]
    # 如果不存在就运行下一个
    outputDict = detectionFromInputDict(configJsonDict)
    endTime1 = time.perf_counter()
    print('Running time: %s Seconds' % (endTime1 - startTime))
