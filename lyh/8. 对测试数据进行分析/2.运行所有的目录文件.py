import os
import sys
import time
from typing import List

from hpc.l3l2utils.L2L3Main import detectionFromInputDict
from hpc.l3l2utils.ParsingJson import readJsonToDict

alldatadirs = [
        R"DATA/测试数据/WRF/1KM",
        R"DATA/测试数据/WRF/3KM",
        R"DATA/测试数据/WRF/9KM",
        R"DATA/测试数据/Grapes/test1",
        R"DATA/测试数据/Grapes/国家超算",
]

def getDirs(dirpaths) -> List[str]:
    dirnamess = os.listdir(dirpaths)
    dirlists = [os.path.join(dirpaths, idir, "centos11") for idir in dirnamess if os.path.exists(os.path.join(dirpaths, idir, "centos11"))]
    dirlists.extend([os.path.join(dirpaths, idir, "centos16") for idir in dirnamess if os.path.exists(os.path.join(dirpaths, idir, "centos16"))])
    # dirlists.extend([os.path.join(dirpaths, idir, "centos21") for idir in dirnamess if os.path.exists(os.path.join(dirpaths, idir, "centos21"))])
    # dirlists.extend([os.path.join(dirpaths, idir, "centos26") for idir in dirnamess if os.path.exists(os.path.join(dirpaths, idir, "centos26"))])
    return dirlists

# 指定某一个文件夹进行运行
substrrundir = None

if __name__ == "__main__":
    startTime = time.perf_counter()
    # configfilepath = R"L2层和L3层结合预测数据/4. 使用输入输出接口对测试数据进行检测/2.检测每个测试文件中的数据/2.config.json"
    configfilepath = os.path.join(sys.path[0], "config.json")
    configJsonDict = readJsonToDict(*(os.path.split(configfilepath)))
    alldatapath = []
    for i in alldatadirs:
        alldatapath.extend(getDirs(i))
    if substrrundir is not None:
       alldatapath = [i for i in alldatapath if substrrundir in i]
    for ipath in alldatapath:
        print(ipath)
        startTime1 = time.perf_counter()
        configJsonDict["predictdirjsonpath"] = os.path.join(ipath, "jsonfile", "alljson.json")
        configJsonDict["spath"] = os.path.join(ipath, "jsonfile", "中间结果生成")
        # 存在中间文件就继续
        if os.path.exists(configJsonDict["spath"]):
            continue
        # 如果不存在就运行下一个
        if not os.path.exists(configJsonDict["predictdirjsonpath"]):
            continue
        if "notrun" in ipath:
            continue
        outputDict = detectionFromInputDict(configJsonDict)
        endTime1 = time.perf_counter()
        print('Running time: %s Seconds' % (endTime1 - startTime1))

    endTime = time.perf_counter()
    print('Running time: %s Seconds' % (endTime - startTime))
