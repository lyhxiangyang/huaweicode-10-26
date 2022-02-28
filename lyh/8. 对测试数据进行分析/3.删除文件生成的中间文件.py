import os
import shutil
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

# 指定某一个文件夹进行
if __name__ == "__main__":
    startTime = time.perf_counter()
    # configfilepath = R"L2层和L3层结合预测数据/4. 使用输入输出接口对测试数据进行检测/2.检测每个测试文件中的数据/2.config.json"
    configfilepath = os.path.join(sys.path[0], "config.json")
    alldatapath = []
    for i in alldatadirs:
        alldatapath.extend(getDirs(i))
    for ipath in alldatapath:
        startTime1 = time.perf_counter()
        deletedir = os.path.join(ipath, "jsonfile", "中间结果生成_90和80合并")
        if os.path.exists(deletedir):
            shutil.rmtree(deletedir)


    endTime = time.perf_counter()
    print('Running time: %s Seconds' % (endTime - startTime))
