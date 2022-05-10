import os
import sys
import time

from hpc.l3l2utils.L2L3Main import detectionFromInputDict
from hpc.l3l2utils.ParsingJson import readJsonToDict


if __name__ == "__main__":
    startTime = time.perf_counter()
    # modelconfigfilepath = R"L2层和L3层结合预测数据/4. 使用输入输出接口对测试数据进行检测/2.检测每个测试文件中的数据/2.config.json"
    configfilepath = os.path.join(sys.path[0], "2.config.json")
    configJsonDict = readJsonToDict(*(os.path.split(configfilepath)))
    outputDict = detectionFromInputDict(configJsonDict)
    endTime = time.perf_counter()
    print('Running time: %s Seconds' % (endTime - startTime))
