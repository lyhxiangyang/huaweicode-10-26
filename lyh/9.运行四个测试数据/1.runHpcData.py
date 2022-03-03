import os
import sys
import time

from hpc.l3l2utils.L2L3Main import detectionFromInputDict
from hpc.l3l2utils.ParsingJson import readJsonToDict

rundirs = [
    R"C:\Users\lWX1084330\Desktop\所有数据整理\四个测试数据单独case\数据\3.1",
    R"C:\Users\lWX1084330\Desktop\所有数据整理\四个测试数据单独case\数据\3.2",
    R"C:\Users\lWX1084330\Desktop\所有数据整理\四个测试数据单独case\数据\3.9",
    R"C:\Users\lWX1084330\Desktop\所有数据整理\四个测试数据单独case\数据\3.10",
]
if __name__ == "__main__":
    configfilepath = os.path.join(sys.path[0], "config.json")
    configJsonDict = readJsonToDict(*(os.path.split(configfilepath)))

    startTime1 = time.perf_counter()
    for irundir in rundirs:
        configjsons = os.listdir(irundir)
        for iconfigjson in configjsons:
            filename = os.path.splitext(iconfigjson)[0]
            configpath = os.path.join(irundir, iconfigjson)
            if not os.path.isfile(configpath):
                continue
            spath = os.path.join(irundir, filename+"_中间结果文件生成")
            if os.path.exists(spath):
                continue
            if not os.path.exists(spath):
                os.mkdir(spath)
            configJsonDict["predictdirjsonpath"] = configpath
            configJsonDict["spath"] = spath
            outputDict = detectionFromInputDict(configJsonDict)
    endTime1 = time.perf_counter()
    print('Running time: %s Seconds' % (endTime1 - startTime1))


