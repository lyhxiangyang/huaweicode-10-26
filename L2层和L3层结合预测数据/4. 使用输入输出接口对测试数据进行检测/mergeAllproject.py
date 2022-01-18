import os
import shutil

"""
将以上所有的项目进行合并
"""


def makealldirexists(dstdir):
    # 先保证dstdir下的网络等目录都存在
    tpath = os.path.join(dstdir, "l2")
    if not os.path.exists(tpath):
        os.makedirs(tpath)
    tpath = os.path.join(dstdir, "network")
    if not os.path.exists(tpath):
        os.makedirs(tpath)
    tpath = os.path.join(dstdir, "ping")
    if not os.path.exists(tpath):
        os.makedirs(tpath)
    tpath = os.path.join(dstdir, "process")
    if not os.path.exists(tpath):
        os.makedirs(tpath)
    tpath = os.path.join(dstdir, "server")
    if not os.path.exists(tpath):
        os.makedirs(tpath)
    tpath = os.path.join(dstdir, "topdown")
    if not os.path.exists(tpath):
        os.makedirs(tpath)


"""
srcdir下面有network、l2、ping网络，dstdir下面有network、l2、ping
"""

def copyallDirToDstDir(srcdir, dstdir, prefixname):
    makealldirexists(dstdir)

    tsrcdir = os.path.join(srcdir, "l2")
    tdstdir = os.path.join(dstdir, "l2")
    for ifile in os.listdir(tsrcdir):
        copyfilename = prefixname+ifile
        shutil.copy(os.path.join(tsrcdir, ifile), os.path.join(tdstdir, copyfilename))

    tsrcdir = os.path.join(srcdir, "network")
    tdstdir = os.path.join(dstdir, "network")
    for ifile in os.listdir(tsrcdir):
        copyfilename = prefixname+ifile
        shutil.copy(os.path.join(tsrcdir, ifile), os.path.join(tdstdir, copyfilename))

    tsrcdir = os.path.join(srcdir, "ping")
    tdstdir = os.path.join(dstdir, "ping")
    for ifile in os.listdir(tsrcdir):
        copyfilename = prefixname+ifile
        shutil.copy(os.path.join(tsrcdir, ifile), os.path.join(tdstdir, copyfilename))

    tsrcdir = os.path.join(srcdir, "process")
    tdstdir = os.path.join(dstdir, "process")
    for ifile in os.listdir(tsrcdir):
        copyfilename = prefixname+ifile
        shutil.copy(os.path.join(tsrcdir, ifile), os.path.join(tdstdir, copyfilename))

    tsrcdir = os.path.join(srcdir, "server")
    tdstdir = os.path.join(dstdir, "server")
    for ifile in os.listdir(tsrcdir):
        copyfilename = prefixname+ifile
        shutil.copy(os.path.join(tsrcdir, ifile), os.path.join(tdstdir, copyfilename))

    tsrcdir = os.path.join(srcdir, "topdown")
    tdstdir = os.path.join(dstdir, "topdown")
    for ifile in os.listdir(tsrcdir):
        copyfilename = prefixname+ifile
        shutil.copy(os.path.join(tsrcdir, ifile), os.path.join(tdstdir, copyfilename))

centos11Series11 = [1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17]
centos16Series16 = [5, 6, 7, 8, 9, 10, 18, 19]

if __name__ == "__main__":
    runscriptpath = R"DATA/2022-01-14新的测试数据/"  # 得到当前脚本的相对路径
    saveDatapath = R"DATA/2022-01-14新的测试数据/20.总数据集合"
    dirs = os.listdir(runscriptpath)
    for idir in dirs:
        # 不是目录就continue
        if not os.path.isdir(os.path.join(runscriptpath, idir)):
            continue
        dirnumber = int(idir.split(".")[0])
        if dirnumber not in centos11Series11 and dirnumber not in centos16Series16:
            continue
        print("拷贝{}".format(idir))
        srcdir = os.path.join(runscriptpath, idir, "centos11")
        if dirnumber in centos16Series16:
            srcdir = os.path.join(runscriptpath, idir, "centos16")
        copyallDirToDstDir(srcdir, saveDatapath, prefixname=idir)

