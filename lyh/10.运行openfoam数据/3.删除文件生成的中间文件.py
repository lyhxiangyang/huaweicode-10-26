import os
import shutil
import sys
import time
from typing import List

from hpc.l3l2utils.L2L3Main import detectionFromInputDict
from hpc.l3l2utils.ParsingJson import readJsonToDict

alldatadirs = [
    R"csvfiles/normals_tmp",
    R"csvfiles/abnormals/allcpu10",
    R"csvfiles/abnormals/bandwidth50_t0",
    R"csvfiles/abnormals/bandwidth50_t1",
    R"csvfiles/abnormals/bandwidth50_t2",
    R"csvfiles/abnormals/cacheGrab90",
    R"csvfiles/abnormals/cpuGrab80",
    R"csvfiles/abnormals/memleak60",
    R"csvfiles/abnormals/singlecpu20",
]

def getDirs(dirpaths) -> List[str]:
    dirnamess = os.listdir(dirpaths)
    dirpaths = [os.path.join(dirpaths, idir) for idir in dirnamess]
    return dirpaths

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
        deletedir = os.path.join(ipath, "jsonfile", "中间结果生成50_80_90_again")
        if os.path.exists(deletedir):
            print(deletedir)
            shutil.rmtree(deletedir)


    endTime = time.perf_counter()
    print('Running time: %s Seconds' % (endTime - startTime))
