import json
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Set

import joblib
import pandas as pd

from Classifiers.ModelPred import select_and_pred
from l3l2utils.DataOperation import pushLabelToFirst
from l3l2utils.DefineData import TIME_COLUMN_NAME, FAULT_FLAG, CPU_FEATURE, MODEL_TYPE, PROCESS_CPUNAME

"""
得到以prefixnames中元素为前缀的所有值
"""


def getTrainedFeatures(dfcolumns: List[str], prefixnames: List[str]):
    if prefixnames is None:
        return dfcolumns
    # resList = []
    # for iprefixname in prefixnames:
    #     for idfcolumn in dfcolumns:
    #         if idfcolumn.startswith(iprefixname):
    #             resList.append(idfcolumn)
    # return resList
    return [idfcolumn for idfcolumn in dfcolumns for iprefixname in prefixnames if
            idfcolumn.startswith(iprefixname)]  # 一句话解决代码


"""
函数功能： 通过server数据和process数据合并之后提取有用信息 包括每时每刻各个核心是否有异常以及server预测需要用到的指标
"""


def detectL3CPUAbnormal(allserverpds: pd.DataFrame, allprocesspds: pd.DataFrame, spath: str = None,
                        modelfilepath: str = None, modeltype=0,
                        processFeatures=None):
    def getcores(processpd: pd.DataFrame) -> Tuple[int, Set[int]]:
        coresSet = set(list(processpd[CPU_FEATURE]))
        coresnum = len(coresSet)
        return coresnum, coresSet

    # ======== detectL3CPUAbnormal运行
    if processFeatures is None:
        processFeatures = ["cpu"]
    # 将allserverpds里面所有的时间搜集起来
    timecolumns = allserverpds[TIME_COLUMN_NAME]
    serverinformationDict = defaultdict(list)
    serverinformationDict[TIME_COLUMN_NAME] = timecolumns  # 加入时间
    for stime in timecolumns:
        # 检测某个时间点下
        detectresult = detectionCPUInPointTime(allprocesspds, stime, modelfilepath=modelfilepath, modeltype=modeltype,
                                               processFeatures=processFeatures)
        wrf_cpu_time = detectresult[0]
        abnormalcores = detectresult[1]
        abnormalcoremaxtime = detectresult[2]
        # 不管返回值如何都进行直接的添加
        serverinformationDict["wrf_cpu"].append(wrf_cpu_time)  # 这一时刻的wrf使用的cpu时间
        serverinformationDict["abnormalcores"].append(abnormalcores)  # 这一时刻使用到的异常核心数
        serverinformationDict["coresmaxtime"].append(abnormalcoremaxtime)  # 这一时刻异常核对应的时间
        serverinformationDict["coresnums"].append(-1 if abnormalcores is None else len(abnormalcores))  # 核心的数量
    if FAULT_FLAG in allserverpds.columns.array:
        serverinformationDict[FAULT_FLAG] = list(allserverpds[FAULT_FLAG])
    wrfruncoresnumber, coresSet = getcores(allprocesspds)
    cpuabnormalList = predictcpu(serverinformationDict, wrfruncoresnumber)
    # 将字典中的数据进行保存 ==========================================================================================
    if spath is not None:
        if not os.path.exists(spath):
            os.makedirs(spath)
        savedict = serverinformationDict
        tpd = pd.DataFrame(data=savedict)
        if FAULT_FLAG in savedict.keys():
            pushLabelToFirst(tpd, FAULT_FLAG)
        pushLabelToFirst(tpd, TIME_COLUMN_NAME)
        tpd.to_csv(os.path.join(spath, "server_process有用指标.csv"))
        # 输出并保存核心数量信息
        print("wrf运行核心数量：{}".format(wrfruncoresnumber))
        print("核心的位数：{}".format(coresSet))
        with open(os.path.join(spath, "运行核心的数据.txt"), "w", encoding="utf-8") as f:
            writeinfo = ["核心数量：{}\n".format(wrfruncoresnumber), "核心的位数：{}\n".format(coresSet)]
            f.writelines(writeinfo)
    # ==============================================================================================================
    return cpuabnormalList


"""
函数功能：预测process某个时刻每个核心的状态
函数返回值：1.当前时刻cpu总值，2.异常CPU列表  3. 异常CPU耗费的时间列表  
"""


def detectionCPUInPointTime(processpds: pd.DataFrame, nowtime: str, modelfilepath: str = None, modeltype=0,
                            processFeatures=None):
    if processFeatures is None:
        processFeatures = ["cpu"]
    nowdf = processpds[processpds[TIME_COLUMN_NAME] == nowtime]
    if len(nowdf) == 0:
        return 0, None, None
    cpuusefulFeatures = getTrainedFeatures(processpds.columns.tolist(), processFeatures)
    # 先得到总的CPUTIME的时间
    cputime = nowdf[PROCESS_CPUNAME].sum()
    # 核的编号
    cores_serialnumber = list(nowdf.loc[:, CPU_FEATURE])
    # 核的cpu时间
    cores_runtimeList = list(nowdf.loc[:, PROCESS_CPUNAME])
    predictflag = select_and_pred(nowdf[cpuusefulFeatures], MODEL_TYPE[modeltype], saved_model_path=modelfilepath)
    predictflag = [True if i != 0 else False for i in predictflag]
    # predictflag为True代表异常， 否则代表这正常
    # 获得异常的核
    assert len(predictflag) == len(cores_serialnumber)
    abnormalcores = [cores_serialnumber[i] for i, flag in enumerate(predictflag) if flag]
    abnormalcoremaxtime = [cores_runtimeList[i] for i, flag in enumerate(predictflag) if flag]
    # 将所有的cputime和不正常的核心数据进行返回
    return cputime, abnormalcores, abnormalcoremaxtime


"""
对CPU进行预测，
返回值是一个列表, 0 代表这个时间段预测为正常，1预测为CPU异常, -1代表是边界，无法进行预测
预测标准是：
10 代表全CPU异常
20 代表单CPU抢占
30 代表多CPU抢占
80 代表随机抢占
-1 代表这个时间没有数据
"""


def predictcpu(serverinformationDict: Dict, coresnumber: int = 0) -> List[int]:
    #  wrfnumList不为None
    wrfnumList = serverinformationDict['abnormalcores']
    assert len(serverinformationDict[TIME_COLUMN_NAME]) == len(wrfnumList)
    iscpulist = []  # 先添加一个数值，最后返回的时候要去掉
    ilastlist = None
    for i, ilist in enumerate(wrfnumList):
        # ===========================
        if ilist is None:
            iscpulist.append(-1)
            ilastlist = None
            continue
        # ========================
        if len(ilist) == 0:
            iscpulist.append(0)
            ilastlist = []
            continue
        # ========================
        if len(ilist) == 1:
            if ilastlist is None:
                iscpulist.append(20)
            elif len(ilastlist) == 0:
                iscpulist.append(20)
            elif len(ilastlist) == 1 and set(ilastlist) == set(ilist):
                iscpulist.append(20)
            elif len(ilastlist) == 1 and set(ilastlist) != set(ilist):
                iscpulist[-1] = 80
                iscpulist.append(80)
            elif len(ilastlist) > 1:
                iscpulist[-1] = 80
                iscpulist.append(80)
            else:
                print("len(list) == 1: 预测cpu出现了不可预知的错误")
                exit(1)
            ilastlist = ilist
            continue
        # =======================
        if len(ilist) == coresnumber:
            if ilastlist is None:
                iscpulist.append(10)
            elif len(ilastlist) == 0:
                iscpulist.append(10)
            elif len(ilastlist) == coresnumber:
                iscpulist.append(10)
            else:
                iscpulist[-1] = 80
                iscpulist.append(80)
            ilastlist = ilist
            continue
        # =======================
        # 现在就是多核心cpu的数据
        if ilastlist is None:
            iscpulist.append(30)
        elif len(ilastlist) == 0:
            iscpulist.append(30)
        elif len(ilastlist) == 1:
            iscpulist[-1] = 80
            iscpulist.append(80)
        elif len(ilastlist) == coresnumber:
            iscpulist[-1] = 80
            iscpulist.append(80)
        elif len(ilastlist) != len(ilist):
            iscpulist[-1] = 80
            iscpulist.append(80)
        elif len(ilastlist) == len(ilist) and set(ilastlist) != set(ilist):
            iscpulist[-1] = 80
            iscpulist.append(80)
        elif len(ilastlist) == len(ilist) and set(ilastlist) == set(ilist):
            iscpulist[-1] = 30
            iscpulist.append(30)
        else:
            print("多核cpu 来到了不可能来到的位置")
            exit(1)
        ilastlist = ilist
    return iscpulist


"""
识别温度数据
"""


def predictTemp(model_path: str, model_type: str, data: pd.DataFrame):
    FANS = [
        'FAN1_F_Speed', "FAN1_R_Speed",
        'FAN2_F_Speed', "FAN2_R_Speed",
        'FAN3_F_Speed', "FAN3_R_Speed",
        'FAN4_F_Speed', "FAN4_R_Speed",
        'FAN5_F_Speed', "FAN5_R_Speed",
        'FAN6_F_Speed', "FAN6_R_Speed",
        'FAN7_F_Speed', "FAN7_R_Speed",
    ]
    TEMPERATURE = [
        'CPU1_Core_Rem', 'CPU2_Core_Rem', 'CPU3_Core_Rem', 'CPU4_Core_Rem',
        'CPU1_MEM_Temp', 'CPU2_MEM_Temp', 'CPU3_MEM_Temp', 'CPU4_MEM_Temp',
    ]

    def get_extended_features(prefix):
        selected = []
        for p in prefix:
            selected.append(p)
            selected.append(p + '_max')
            selected.append(p + '_min')
            selected.append(p + '_mean')
            selected.append(p + '_percentage50')
        return selected

    result = []
    for i, temp in enumerate(TEMPERATURE):
        for j, fan in enumerate(FANS):
            extended_features = get_extended_features(['freq', temp, fan])
            select_data = data[extended_features]
            model = joblib.load(os.path.join(model_path, model_type + '.pkl'))
            y = model.predict(select_data)
            if i == 0 and j == 0:
                result = y
            for k, v in enumerate(y):
                if v == 3:
                    if result[k] == 0:
                        result[k] = 3
                if v == 4:
                    if result[k] == 0 or result[k] == 3:
                        result[k] = 4
    return result


"""
使用内存泄露模型进行预测
"""


def detectL3MemLeakAbnormal(allserverpds: pd.DataFrame, modelfilepath: str = None, modeltype=0, serverFeatures=None):
    memleakUserfulFeatures = getTrainedFeatures(allserverpds.columns.tolist(), serverFeatures)
    testPd = allserverpds[memleakUserfulFeatures]
    memleakPreFlagList = select_and_pred(testPd, MODEL_TYPE[modeltype], saved_model_path=modelfilepath)
    return memleakPreFlagList


"""
使用内存带宽模型进行预测
"""


def detectL3BandWidthAbnormal(allserverpds: pd.DataFrame, modelfilepath: str = None, modeltype=0, serverFeatures=None):
    bandwidthUserfulFeatures = getTrainedFeatures(allserverpds.columns.tolist(), serverFeatures)
    testPd = allserverpds[bandwidthUserfulFeatures]
    bandwidthPreFlagList = select_and_pred(testPd, MODEL_TYPE[modeltype], saved_model_path=modelfilepath)
    return bandwidthPreFlagList

"""
检测网络异常情况 TXHang
"""
def detectNetwork_TXHangAbnormal(allnetworkpds: pd.DataFrame, isExistFlag: bool = True):
    threshold_avg_lat = 100
    data = allnetworkpds.groupby(TIME_COLUMN_NAME, as_index=False).agg([max])
    prenet = []
    for i in data['avg_lat'].tolist():
        if i > threshold_avg_lat:
            prenet.append(151)
        else:
            prenet.append(0)
    result = data[TIME_COLUMN_NAME].to_frame()
    if isExistFlag:
        result[FAULT_FLAG] = data[FAULT_FLAG]
    result['preFLag'] = prenet
    # result.set_index(TIME_COLUMN_NAME, inplace=True)
    return result
