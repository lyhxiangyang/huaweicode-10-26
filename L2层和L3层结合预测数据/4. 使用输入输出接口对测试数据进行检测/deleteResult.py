import os
import shutil
import subprocess
import sys
import time

"""
脚本作用自动运行1-19中的脚本
"""
runSeries = list(range(1, 20))
if __name__ == "__main__":
    runscriptpath = R"DATA/2022-01-14新的测试数据/"  # 得到当前脚本的相对路径
    dirs = os.listdir(runscriptpath)
    for idir in dirs:
        # 不是目录就continue
        if not os.path.isdir(os.path.join(runscriptpath, idir)):
            continue
        dirnumber = int(idir.split(".")[0])
        if dirnumber not in runSeries:
            continue
        nowpath = os.path.join(runscriptpath, idir)
        deletepath = os.path.join(nowpath, "centos11", "jsonfile")
        if os.path.exists(deletepath):
            print("删除: {}".format(deletepath))
            shutil.rmtree(deletepath)
        deletepath = os.path.join(nowpath, "centos16", "jsonfile")
        if os.path.exists(deletepath):
            print("删除: {}".format(deletepath))
            shutil.rmtree(deletepath)

        deletepath = os.path.join(nowpath, "centos21", "jsonfile")
        if os.path.exists(deletepath):
            print("删除: {}".format(deletepath))
            shutil.rmtree(deletepath)
        deletepath = os.path.join(nowpath, "centos26", "jsonfile")
        if os.path.exists(deletepath):
            print("删除: {}".format(deletepath))
            shutil.rmtree(deletepath)
