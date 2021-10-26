import os
from collections import defaultdict
from typing import Dict, Tuple, Union, Any, List

import pandas as pd

"""
-   文件目的：从路径下读取对应的核信息，
readpath： 读取的路径
excludecore: 要排除读取信息的核
"""


# 读取某个单独的错误码
# readpath指的是 包含1.csv的目录
def readCoresPD(readpath: str, excludecore=None, select_feature: List[str] = None) -> Union[
    Tuple[None, bool], Tuple[dict[int, Any], bool]]:
    if excludecore is None:
        excludecore = []
    if not os.path.exists(readpath):
        return None, True
    core_pd_Dict = {}
    for score in os.listdir(readpath):
        icore = int(os.path.splitext(score)[0])
        if icore in excludecore:
            continue
        tpathfile = os.path.join(readpath, score)
        tpd = pd.read_csv(tpathfile)
        if select_feature is not None:
            tpd = tpd[select_feature]
        core_pd_Dict[icore] = tpd
    return core_pd_Dict, False


# 将readpath路径下的所有错误码进行读取，排除excludeFaulty，readDir是每个错误码下的读取目录
# 返回一个字典 faulty-core-DataFrame
def readFaultyPD(readpath: str, readDir: str, excludeFaulty=None, select_feature=None) -> Union[
    Tuple[None, bool], Tuple[defaultdict[Any, Dict], bool]]:
    if select_feature is None:
        select_feature = []
    if excludeFaulty is None:
        excludeFaulty = []
    if not os.path.exists(readpath):
        return None, True
    faultDict = defaultdict(dict)
    for strFaulty in os.listdir(readpath):
        ifaulty = int(str.split(strFaulty, "_")[1])
        if ifaulty in excludeFaulty:
            continue
        tpath = os.path.join(readpath, strFaulty, readDir)
        if not os.path.exists(tpath):
            print("{} 不存在".format(tpath))
            continue
        tdict, err = readCoresPD(tpath, select_feature=select_feature)
        if err:
            print("{}核心读取失败".format(ifaulty))
            exit(1)
        faultDict[ifaulty] = tdict
    return faultDict, False


# 将int - DataFrame这种格式的字典保存
# 将会在savepath目录下生成一系列的 0.csv的文件
def saveFaultyDict(savepath: str, faultydict: Dict[int, pd.DataFrame]):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    for ifaulty, ifaultypd in faultydict.items():
        tfilepath = os.path.join(savepath, str(ifaulty) + ".csv")
        ifaultypd.to_csv(tfilepath, index=False)


def readFaultyDict(savepath: str):
    faultydict = {}
    files = os.listdir(savepath)
    for ifile in files:
        tfile = int(os.path.splitext(ifile)[0])
        tpathfile = os.path.join(savepath, ifile)
        tpd = pd.read_csv(tpathfile)
        faultydict[tfile] = tpd
    return faultydict


# 将int-int-DataFrame这种格式的字典进行保存
def saveFaultyCoreDict(savepath: str, faulty_core_dict: Dict[int, Dict[int, pd.DataFrame]]):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    for ifault, icoredict in faulty_core_dict.items():
        tfaultpath = os.path.join(savepath, str(ifault))
        if not os.path.exists(tfaultpath):
            os.makedirs(tfaultpath)
        for icore, ipd in icoredict.items():
            tfaultcorefile = os.path.join(tfaultpath, str(icore) + ".csv")
            ipd.to_csv(tfaultcorefile, index=False)


def readFaultyCoreDict(savepath: str) -> Dict[int, Dict[int, pd.DataFrame]]:
    if not os.path.exists(savepath):
        print("读取路径不存在")
        exit(1)
    faultydirs = os.listdir(savepath)
    faulty_core_dict = defaultdict(dict)
    for sfault in faultydirs:
        ifault = int(sfault)
        tfaultpath = os.path.join(savepath, sfault)
        corefiles = os.listdir(tfaultpath)
        for scorefile in corefiles:
            icorename = int(os.path.splitext(scorefile)[0])
            tfaultcorefile = os.path.join(tfaultpath, scorefile)
            faulty_core_dict[ifault][icorename] = pd.read_csv(tfaultcorefile)
    return faulty_core_dict


# 保存 str-int-int类型
# 文件-时间段-核心PD

def saveFilename_Time_Core_pdDict(savepath: str, ftcPD: Dict):
    for filename, time_core_pdDict in ftcPD.items():
        for time, core_pdDict in time_core_pdDict.items():
            for icore, tpd in core_pdDict.items():
                tpath = os.path.join(savepath, filename, str(time))
                tfilename = os.path.join(tpath, str(icore) + ".csv")
                if not os.path.exists(tpath):
                    os.makedirs(tpath)
                tpd.to_csv(tfilename, index=False)


def readFilename_Time_Core_pdDict(readpath: str) -> Dict:
    filename_time_corePd = {}
    filenames = os.listdir(readpath)
    for istrfilename in filenames:
        filepath = os.path.join(readpath, istrfilename)
        times = os.listdir(filepath)
        filename_time_corePd[istrfilename] = {}
        for istrtime in times:
            file_timepath = os.path.join(filepath, istrtime)
            itime = int(istrtime)
            coresname = os.listdir(file_timepath)
            filename_time_corePd[istrfilename][itime] = {}
            for istrcorename in coresname:
                score = os.path.splitext(istrcorename)[0]
                icore = int(score)
                file_time_corepath = os.path.join(file_timepath, istrcorename)
                filename_time_corePd[istrfilename][itime][icore] = pd.read_csv(file_time_corepath)
    return filename_time_corePd


# 保存 str-int-int-int类型
# 文件-时间段-核心-错误PD

def saveFilename_Time_Core_Faulty_pdDict(savepath: str, ftcPD: Dict):
    for filename, time_core_pdDict in ftcPD.items():
        for time, core_pdDict in time_core_pdDict.items():
            for icore, faultypdDict in core_pdDict.items():
                for ifault, tpd in faultypdDict.items():
                    tpath = os.path.join(savepath, filename, str(time), str(icore))
                    tfilename = os.path.join(tpath, str(ifault) + ".csv")
                    if not os.path.exists(tpath):
                        os.makedirs(tpath)
                    tpd.to_csv(tfilename, index=False)


def readFilename_Time_Core_Faulty_pdDict(readpath: str) -> Dict:
    filename_time_core_faultPd = {}
    filenames = os.listdir(readpath)
    for istrfilename in filenames:
        filepath = os.path.join(readpath, istrfilename)
        times = os.listdir(filepath)
        filename_time_core_faultPd[istrfilename] = {}
        for istrtime in times:
            file_timepath = os.path.join(filepath, istrtime)
            itime = int(istrtime)
            coresname = os.listdir(file_timepath)
            filename_time_core_faultPd[istrfilename][itime] = {}
            for istrcore in coresname:
                icore = int(istrcore)
                file_time_corepath = os.path.join(file_timepath, istrcore)
                filename_time_core_faultPd[istrfilename][itime][icore] = {}
                faultys = os.listdir(file_time_corepath)
                for istrfaultyname in faultys:
                    sfaulty = os.path.splitext(istrfaultyname)[0]
                    ifaulty = int(sfaulty)
                    file_timr_core_faultpath = os.path.join(file_time_corepath, istrfaultyname)
                    filename_time_core_faultPd[istrfilename][itime][icore][ifaulty] = pd.read_csv(file_timr_core_faultpath)
    return filename_time_core_faultPd
