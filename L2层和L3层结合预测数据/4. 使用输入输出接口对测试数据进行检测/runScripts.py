import os
import subprocess
import sys
import time

"""
脚本作用自动运行1-19中的脚本
"""
runSeries = list(range(1, 20))
result1 = []
result2 = []

"""
判断centos11 centos16 centos21 centos26是否在目录中
true 代表有目录
"""
def judgeHaveDirectory(dirpath):
    dirs = [idir for idir in os.listdir(dirpath) if os.path.isdir(idir)]
    return len(dirs) != 0, dirs


def runFunctions(runscriptpath, idir):
    filepath = os.path.join(runscriptpath, idir, "1.generateJson.py")
    scriptorder1 = [R"C:\Users\lWX1084330\AppData\Local\Programs\Python\Python39\python.exe ", R"{}".format(filepath)]
    with subprocess.Popen(scriptorder1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        while p.poll() is None:
            line = p.stdout.readline().strip().decode("utf-8")
            if len(line) == 0:
                time.sleep(1)
                continue
            print(line)
        stat = p.poll()
        if stat == 0:
            result1.append("处理-{} 成功运行".format(filepath))
        else:
            line = p.stderr.readline().strip()
            print(line)
            result1.append("处理-{} 异常退出".format(filepath))
            exit(1)
    filepath = os.path.join(runscriptpath, idir, "2.L2L3输入输出接口.py")
    scriptorder1 = [R"C:\Users\lWX1084330\AppData\Local\Programs\Python\Python39\python.exe ", R"{}".format(filepath)]
    with subprocess.Popen(scriptorder1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        while p.poll() is None:
            line = p.stdout.readline().strip().decode("utf-8")
            if len(line) == 0:
                time.sleep(1)
                continue
            print(line)
        stat = p.poll()
        if stat == 0:
            result1.append("处理-{} 成功运行".format(filepath))
        else:
            line = p.stderr.readline().strip()
            print(line)
            result1.append("处理-{} 异常退出".format(filepath))
            exit(1)

if __name__ == "__main__":
    runscriptpath = os.path.join(sys.path[0])  # 得到当前脚本的相对路径
    dirs = os.listdir(runscriptpath)
    for idir in dirs:
        # 不是目录就continue
        if not os.path.isdir(os.path.join(runscriptpath, idir)):
            continue
        dirnumber = int(idir.split(".")[0])
        if dirnumber not in runSeries:
            result1.append("处理-----{} 跳过".format(idir))
            continue

        print("处理{}中".format(idir).center(50, "#"))

        issubdir, subdirs = judgeHaveDirectory(os.path.join(runscriptpath,idir))
        if issubdir:
            for isubdir in subdirs:
                runFunctions(os.path.join(runscriptpath, idir), isubdir)
            continue
        # 没有子目录
        runFunctions(runscriptpath, idir)

    # 输出每个文件的最终结果
    print("\n".join(result1))
