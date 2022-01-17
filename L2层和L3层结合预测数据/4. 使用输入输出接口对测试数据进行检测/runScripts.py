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
if __name__ == "__main__":
    runscriptpath = os.path.join(sys.path[0])  # 得到当前脚本的相对路径
    dirs = os.listdir(runscriptpath)
    for idir in dirs:
        dirnumber = int(idir.split(".")[0])
        if dirnumber not in runSeries:
            result1.append("处理-----{} 跳过".format(idir))
            continue

        print("处理{}中".format(idir).center(50, "#"))

        tpath = os.path.join(runscriptpath, idir, "1.generateJson.py").replace(' ', '\ ')
        scriptorder1 = R"C:\Users\lWX1084330\AppData\Local\Programs\Python\Python39\python.exe" + tpath
        with subprocess.Popen(scriptorder1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
            while p.poll() is None:
                line = p.stdout.readline().strip()
                if len(line) == 0:
                    time.sleep(1)
                    continue
                print(line)
            stat = p.poll()
            if stat == 0:
                result1.append("处理-{} 成功运行".format(idir))
            else:
                line = p.stderr.readline().strip()
                print(line)
                result1.append("处理-{} 异常退出".format(idir))
                break

        tpath = os.path.join(runscriptpath, idir, "2.L2L3输入输出接口.py").replace(' ', '\ ')
        scriptorder1 = R"C:\Users\lWX1084330\AppData\Local\Programs\Python\Python39\python.exe" + tpath
        with subprocess.Popen(scriptorder1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
            while p.poll() is None:
                line = p.stdout.readline().strip()
                if len(line) == 0:
                    time.sleep(1)
                    continue
                print(line)
            stat = p.poll()
            if stat == 0:
                result1.append("处理-{} 成功运行".format(idir))
                break
            else:
                line = p.stderr.readline().strip()
                print(line)
                result1.append("处理-{} 异常退出".format(idir))
                break

    # 输出每个文件的最终结果
    print("\n".join(result1))
