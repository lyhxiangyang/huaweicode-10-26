import os
import subprocess
import sys
import time

"""
脚本作用自动运行1-19中的脚本
"""
runSeries = list(range(1, 20))
if __name__ == "__main__":
    runscriptpath = os.path.join(sys.path[0]) # 得到当前脚本的相对路径
    dirs = os.listdir(runscriptpath)
    for idir in dirs:
        print("处理-{}".format(idir), end=" ")
        dirnumber = int(idir.split(".")[0])
        if dirnumber not in runSeries:
            print("跳过")
            continue

        scriptorder1 = "python3 " + os.path.join(runscriptpath, idir, "1.generateJson.py")
        with subprocess.Popen(scriptorder1, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
            p.communicate()
            time.sleep(0.01)
            stat = p.poll()
            if stat != 0:
                print("运行异常")
                break
            print("运行成功")





