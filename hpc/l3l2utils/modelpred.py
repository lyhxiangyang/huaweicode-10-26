import os
from collections import defaultdict
from typing import List, Dict, Tuple, Set

import joblib
import pandas as pd

from hpc.classifiers.ModelPred import select_and_pred
from hpc.l3l2utils.DataFrameOperation import mergeProceeDF, smoothseries, meansmoothseries
from hpc.l3l2utils.DataFrameSaveRead import savepdfile
from hpc.l3l2utils.DataOperation import pushLabelToFirst, getRunHPCTimepdsFromProcess, getsametimepd
from hpc.l3l2utils.DefineData import TIME_COLUMN_NAME, FAULT_FLAG, CPU_FEATURE, MODEL_TYPE, PROCESS_CPUNAME
from hpc.l3l2utils.ParsingJson import getNormalTopdownMean, getNormalServerMean

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
                        modelfilepath: str = None, modeltype=0,):
    def getcores(processpd: pd.DataFrame) -> Tuple[int, Set[int]]:
        coresSet = set(list(processpd[CPU_FEATURE]))
        coresnum = len(coresSet)
        return coresnum, coresSet

    # ======== detectL3CPUAbnormal运行
    # 将allserverpds里面所有的时间搜集起来
    allserverpds = allprocesspds[allprocesspds["cpu_affinity"] == 0]
    timecolumns = allserverpds[TIME_COLUMN_NAME]
    serverinformationDict = defaultdict(list)
    serverinformationDict[TIME_COLUMN_NAME] = timecolumns  # 加入时间
    for stime in timecolumns:
        # 检测某个时间点下
        detectresult = detectionCPUInPointTime(allprocesspds, stime, modelfilepath=modelfilepath, modeltype=modeltype)
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


def detectionCPUInPointTime(processpds: pd.DataFrame, nowtime: str, modelfilepath: str = None, modeltype=0,):
    nowdf = processpds[processpds[TIME_COLUMN_NAME] == nowtime] # 包含了process各个核心的值
    if len(nowdf) == 0:
        return 0, None, None
    # 先得到总的CPUTIME的时间
    cputime = nowdf[PROCESS_CPUNAME].sum()
    # 核的编号
    cores_serialnumber = list(nowdf.loc[:, CPU_FEATURE])
    # 核的cpu时间
    cores_runtimeList = list(nowdf.loc[:, PROCESS_CPUNAME]) # 每个核的CPU时间
    predictflag = select_and_pred(nowdf, MODEL_TYPE[modeltype], saved_model_path=modelfilepath)
    predictflag = [True if i != 0 else False for i in predictflag] # 非0就是异常
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
        if ilist is None: # 表明这个时间没有wrf进程在运行
            iscpulist.append(-1)
            ilastlist = None
            continue
        # ========================
        if len(ilist) == 0: # 表明这个时间没有核心被抢占
            iscpulist.append(0)
            ilastlist = []
            continue
        # ========================
        if len(ilist) == 1: # 只有一个核心被抢占
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
        # if len(ilist) == coresnumber: # 现在就是全核心抢占了
        #     if ilastlist is None:
        #         iscpulist.append(10)
        #     elif len(ilastlist) == 0:
        #         iscpulist.append(10)
        #     elif len(ilastlist) == coresnumber:
        #         iscpulist.append(10)
        #     else:
        #         iscpulist[-1] = 80
        #         iscpulist.append(80)
        #     ilastlist = ilist
        #     continue
        # 如果
        if len(ilist) >= coresnumber // 2: # 现在就是全核心抢占 判断是全核心抢占还是随即抢占
            if ilastlist is None: # 上一个cpu不存在
                iscpulist.append(10)
            elif len(ilastlist) == 0: # 上一个是正常
                iscpulist.append(10)
            elif len(ilastlist) >= coresnumber // 2 and set(ilastlist) == set(ilist): # 上一个指标大于一半 并且和现在的列表一样
                iscpulist.append(10)
            elif len(ilastlist) >= coresnumber // 2  and set(ilastlist) != set(ilist): # 上一个指标的数量大于一半，却和现在的列表不是一个核
                iscpulist[-1] = 80
                iscpulist.append(80)
            else: # 介于1-一半之间
                iscpulist[-1] = 80
                iscpulist.append(80)
            ilastlist = ilist
            continue
        # =======================
        # 现在就是多核心cpu的数据  判断是多CPU还是随机CPU
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
    FANSFeatures = [
        "fan1_speed",
        "fan2_speed",
        "fan3_speed",
        "fan4_speed",
    ]
    TEMPERATUREFeatures = [
        "cpu1_core_rem", "cpu2_core_rem", "cpu3_core_rem", "cpu4_core_rem",
        "cpu1_mem_temp", "cpu2_mem_temp", "cpu3_mem_temp", "cpu4_mem_temp",
        "pch_temp",
    ]
    # FANSFeatures = getTrainedFeatures(data.columns.tolist(), ["FAN"])
    # TEMPERATUREFeatures = getTrainedFeatures(data.columns.tolist(), ["CPU"])


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
    for i, temp in enumerate(TEMPERATUREFeatures):
        for j, fan in enumerate(FANSFeatures):
            extended_features = get_extended_features(["freq", temp, fan])
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
传入的参数：server和process
返回的是带有TIME preFlag 和 FaultFlag的预测结果
"""


def detectL3MemLeakAbnormal(allserverpds: pd.DataFrame,allprocesspd: pd.DataFrame, inputDict: Dict = None):
    # 保证allserverpds和processpds按照时间顺序排列


    modelfilepath = inputDict["servermemory_modelpath"]
    modeltype = inputDict["servermemory_modeltype"]
    memleakpermin=inputDict["memleakpermin"]

    # 对内存的差值处理必须根据进程pid号进行处理
    # 保证内存的指标和pid号的指标长度是一样的
    def diffmemoryseries(memseries: pd.Series, pidseries: pd.Series):
        assert len(memseries) == len(pidseries)
        df = pd.DataFrame(data={
            "mem": memseries,
            "pid": pidseries
        })
        # 直接写个for循环
        reslists = []
        winsize = 5
        for ipid, idf in df.groupby("pid", sort=False):
            other_mem_smooth = smoothseries(idf["mem"], windows=winsize)
            other_mem_smooth_diff = other_mem_smooth.diff(1).fillna(0)
            reslists.extend(other_mem_smooth_diff.tolist())

        return pd.Series(data=reslists)


    # 根据server和process中的memory数据得到内存的变化量
    def getMemory(serverpd: pd.DataFrame, processpd: pd.DataFrame)->pd.DataFrame:
        mergeprocesspd = mergeProceeDF(processpd, sumFeatures=["rss"])
        # 将两者合并
        pspd = pd.merge(left=serverpd, right=mergeprocesspd, left_on=TIME_COLUMN_NAME, right_on=TIME_COLUMN_NAME, suffixes=("", "_y"))

        pspd.fillna(0, inplace=True) # 认为进程不在的时候其数据为0

        servermem = pspd["mem_used"]
        processmem = pspd["rss"]
        othermem = servermem - processmem

        othermemdiff=diffmemoryseries(othermem, pspd["pid"]) / 1000000
        # 返回将带有时间与内存
        respd = pd.DataFrame()
        respd[TIME_COLUMN_NAME] = pspd[TIME_COLUMN_NAME]
        respd["mem_used_mean"] = othermemdiff
        if inputDict["isExistFaultFlag"]:
            respd[FAULT_FLAG] = pspd[FAULT_FLAG]
        return respd
    memorypd = getMemory(serverpd=allserverpds, processpd=allprocesspd)
    memleakPreFlagList = select_and_pred(memorypd, MODEL_TYPE[modeltype], saved_model_path=modelfilepath)

    respd = pd.DataFrame()
    respd[TIME_COLUMN_NAME] = memorypd[TIME_COLUMN_NAME]
    respd["preFlag"] = memleakPreFlagList
    if inputDict["isExistFaultFlag"]:
        respd[FAULT_FLAG] = memorypd[FAULT_FLAG]
    return respd


"""
使用内存带宽模型进行预测
"""


def detectL3BandWidthAbnormal(allserverpds: pd.DataFrame, modelfilepath: str = None, modeltype=0):
    testPd = allserverpds
    bandwidthPreFlagList = select_and_pred(testPd, MODEL_TYPE[modeltype], saved_model_path=modelfilepath)
    return bandwidthPreFlagList

## 传入server和topdown数据，对其进行处理，主要处理操作是对pgfree进行补偿性操作
# process最重要的作用是确定server的起始位置
def detectL3BandWidthAbnormal1(allserverpds: pd.DataFrame, alltopdownpds: pd.DataFrame,allprocesspds: pd.DataFrame, inputDict: Dict = None, detectionJson: Dict = None):
     # 保证iserverpds和itodownpds时间与时间相互匹配
    def compensatePgfree(iserverpd: pd.DataFrame, itopdowndpd: pd.DataFrame, detectionJson: Dict, inplace=True):
        assert len(iserverpd) == len(itopdowndpd)
        if inplace:
            iserverpd = iserverpd.copy()
            itopdowndpd = itopdowndpd.copy()
        # 对itopdownpd中的mflops进行平滑处理
        cname = "mflops"
        itopdowndpd[cname] = itopdowndpd[cname].rolling(window=5, center=True, min_periods=1).median() # 先将最大最小值去除
        itopdowndpd[cname] = itopdowndpd[cname].rolling(window=5, center=True, min_periods=1).mean()
        mflops_mean = getNormalTopdownMean(detectionJson, [itopdowndpd], [cname], datanumber=10)[cname]
        mflops_change = itopdowndpd[cname].apply(lambda x: (mflops_mean - x) / mflops_mean if x < mflops_mean else 0)

        cname = "pgfree"
        iserverpd[cname] = iserverpd[cname].rolling(window=5, center=True, min_periods=1).median() # 先将最大最小值去除
        iserverpd[cname] = iserverpd[cname].rolling(window=5, center=True, min_periods=1).median() # 多去一次
        iserverpd[cname] = iserverpd[cname].rolling(window=5, center=True, min_periods=1).mean()
        pgfree_mean = getNormalServerMean(detectionJson, [iserverpd], [cname], datanumber=10)[cname]
        iserverpd[cname] = iserverpd[cname] + pgfree_mean * mflops_change
        iserverpd[cname] = iserverpd[cname].rolling(window=5, center=True, min_periods=1).median() # 对pgfree得到的结果重新去掉最大值最小值
        # pgfree 需要减去平均值
        iserverpd[cname] = iserverpd[cname] - pgfree_mean
        return iserverpd

    modelfilepath = inputDict["serverbandwidth_modelpath"]
    modeltype = inputDict["serverbandwidth_modeltype"]
    # 将server和process的时间进行对齐，也就是去除HPC之外的时间
    allserverpds = getRunHPCTimepdsFromProcess([allserverpds], [allprocesspds])[0]
    alltopdownpds = getRunHPCTimepdsFromProcess([alltopdownpds], [allprocesspds])[0]
    allserverpds, alltopdownpds = getsametimepd(allserverpds, alltopdownpds)
    testPd = compensatePgfree(allserverpds, alltopdownpds, detectionJson, False)
    # 进行预测
    bandwidthPreFlagList = select_and_pred(testPd, MODEL_TYPE[modeltype], saved_model_path=modelfilepath)


    respd = pd.DataFrame()
    respd[TIME_COLUMN_NAME] = allserverpds[TIME_COLUMN_NAME]
    respd["preFlag"] = bandwidthPreFlagList
    if inputDict["isExistFaultFlag"]:
        respd[FAULT_FLAG] = allserverpds[FAULT_FLAG]

    return respd


"""
检测网络异常情况 TXHang
传入时间按照s进行传入，这个s会经过时间处理，也就是，会进行按照时间的分钟的分组合并，最后返回
"""


def detectNetwork_TXHangAbnormal(allnetworkpds: pd.DataFrame, isExistFlag: bool = True):
    threshold_avg_lat = 100
    data = allnetworkpds.groupby(TIME_COLUMN_NAME, as_index=False).agg("max")
    prenet = []
    for i in data['avg_lat'].tolist():
        if i > threshold_avg_lat:
            prenet.append(151)
        else:
            prenet.append(0)
    result = data[TIME_COLUMN_NAME].to_frame()
    if isExistFlag:
        result[FAULT_FLAG] = data[FAULT_FLAG]
    result['preFlag'] = prenet
    # result1.set_index(TIME_COLUMN_NAME, inplace=True)
    return result

"""
预测机柜功率封顶导致的121异常
"""
def predictCabinet_PowerCapping(model_path: str, model_type: str, l2_serverdata: pd.DataFrame):
    select_data = l2_serverdata[["cabinet_power"]]
    freq = l2_serverdata["freq"].tolist()
    model = joblib.load(os.path.join(model_path, model_type + ".pkl"))
    result = model.predict(select_data)
    for i in range(len(result)):
        if freq[i] > 90:
            result[i] = 0
    return result


"""
预测机器功率封顶导致的111异常
需要把121 131结果作为参数传入进来
"""

def predictServer_PowerCapping(model_path: str, model_type: str, l2_serverdata: pd.DataFrame, resultPds: List[pd.DataFrame]):
    select_data = l2_serverdata[["power"]]
    freq = l2_serverdata["freq"].tolist()
    model = joblib.load(os.path.join(model_path, model_type + ".pkl"))
    result = model.predict(select_data)
    for i in range(len(result)):
        if freq[i] > 90 or result[i] != 111:
            result[i] = 0
    for pd in resultPds:
        res_list = pd["preFlag"].tolist()
        for i in range(len(res_list)):
            if result[i] == 111 and res_list[i] != 0:
                result[i] = 0
    return result


"""
预测CPU异常下降导致的161异常
需要传入121 131结果
"""
def predictL2_CPUDown(model_path: str, model_type: str, l2_serverdata: pd.DataFrame, resultPds: List[pd.DataFrame])-> List:
    select_data = l2_serverdata[["power"]]
    freq = l2_serverdata["freq"].tolist()
    model = joblib.load(os.path.join(model_path, model_type + ".pkl"))
    result = model.predict(select_data)
    for i in range(len(result)):
        if freq[i] > 90 or result[i] != 161:
            result[i] = 0
    for pd in resultPds:
        res_list = pd["preFlag"].tolist()
        for i in range(len(res_list)):
            if result[i] == 161 and res_list[i] != 0:
                result[i] = 0
    return result

"""
预测是否为Cache抢占，由于cachegrab模型的作用是用来判断是否是50和90，所以还需要将50的结果传入进来
"""
def predictCacheGrab(l2_serverdata: pd.DataFrame,bandwidthResult: pd.DataFrame, modelfilepath: str = None, modeltype=0)->List:
    bandwidthrList = bandwidthResult["preFlag"].tolist()
    rd_wr_sumList = select_and_pred(l2_serverdata, MODEL_TYPE[modeltype], saved_model_path=modelfilepath)
    assert len(rd_wr_sumList) == len(bandwidthrList)
    resList = []
    for i in range(0, len(rd_wr_sumList)):
        if rd_wr_sumList[i] == 0:
            resList.append(0)
            continue
        # 现在预测为50或90
        if bandwidthrList[i] == 50:
            resList.append(0) # 现在是50了，那代表不是90
            continue
        resList.append(90)
    return resList

# 对数据进行处理
def predictCacheGrab1(alltopdownpds: pd.DataFrame, bandwidthResult: pd.DataFrame,inputDict: Dict, detectJsonDict: Dict):
    # 对读写操作进行补偿措施
     # 保证iserverpds和itodownpds时间与时间相互匹配
    def compensateRW(itopdownpd: pd.DataFrame, detectJson: Dict, inplace=True):
        if inplace:
            itopdownpd = itopdownpd.copy()
        # 对itopdownpd中的mflops进行平滑处理
        cname = "mflops"
        # itopdownpd = removeUselessDataFromTopdownList([itopdownpd])[0]
        itopdownpd[cname] = itopdownpd[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
        itopdownpd[cname] = itopdownpd[cname].rolling(window=5, center=True, min_periods=1).mean()
        mflops_mean = getNormalTopdownMean(detectJson, [itopdownpd], [cname], datanumber=10)[cname]
        mflops_change = itopdownpd[cname].apply(lambda x: (mflops_mean - x) / mflops_mean if x < mflops_mean else 0)
        itopdownpd["mflops_change"] = mflops_change
        mflops_change = itopdownpd["mflops_change"]

        # 对ddrc_rd进行滑动窗口处理
        rd_cname = "ddrc_rd"
        itopdownpd[rd_cname] = itopdownpd[rd_cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
        itopdownpd[rd_cname] = itopdownpd[rd_cname].rolling(window=5, center=True, min_periods=1).mean()
        ddrc_rd_mean = getNormalTopdownMean(detectJson, [itopdownpd], [rd_cname], datanumber=10)[rd_cname]
        itopdownpd[rd_cname] = itopdownpd[rd_cname] + ddrc_rd_mean * mflops_change

        # 对ddrc_rd进行滑动窗口处理
        wr_cname = "ddrc_wr"
        itopdownpd[wr_cname] = itopdownpd[wr_cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
        itopdownpd[wr_cname] = itopdownpd[wr_cname].rolling(window=5, center=True, min_periods=1).mean()
        ddrc_rd_mean = getNormalTopdownMean(detectJson, [itopdownpd], [wr_cname], datanumber=10)[wr_cname]
        itopdownpd[wr_cname] = itopdownpd[wr_cname] + ddrc_rd_mean * mflops_change

        # 对rd_wr_sum进行结合 减去平均值  阈值与6000比较
        rd_wr_cname = "ddrc_ddwr_sum"
        itopdownpd[rd_wr_cname] = itopdownpd[rd_cname] + itopdownpd[wr_cname]
        itopdownpd[rd_wr_cname] = itopdownpd[rd_wr_cname].rolling(window=5, center=True, min_periods=1).median()
        rd_wr_sum_mean = getNormalTopdownMean(detectJson, [itopdownpd], [rd_wr_cname], datanumber=10)[rd_wr_cname]
        itopdownpd[rd_wr_cname] = itopdownpd[rd_wr_cname] - rd_wr_sum_mean
        # 重点是mflops、ddrc_rd、ddrc_ddwr_sum
        return itopdownpd

    # 先把必要的提取
    modeltype = inputDict["cachegrab_modeltype"]
    modelfilepath =  inputDict["cachegrab_modelpath"]


    ttopdownpd = compensateRW(alltopdownpds, detectJsonDict)
    if inputDict["spath"] is not None:
        tpath = os.path.join(inputDict["spath"], "abnormalInfo", "cacheGrab")
        savepdfile(ttopdownpd, spath=tpath, filename="topdown.csv")

    rd_wr_sumList = select_and_pred(ttopdownpd, MODEL_TYPE[modeltype], saved_model_path=modelfilepath)

    bandwidthResult = bandwidthResult.set_index("time")
    cacheResult = pd.Series(index=ttopdownpd[TIME_COLUMN_NAME], data=rd_wr_sumList)

    resList = []
    for itime, ivalue in cacheResult.items():
        if itime not in bandwidthResult:
            resList.append(0)
            continue
        if ivalue == 0:
            resList.append(0)
            continue
        # 那么现在预测不为0，肯定是有异常与50进行区别
        if bandwidthResult[itime] == 50:
            resList.append(0)
            continue
        resList.append(90)

    restpd = pd.DataFrame()
    restpd[TIME_COLUMN_NAME] = ttopdownpd[TIME_COLUMN_NAME]
    if inputDict["isExistFaultFlag"]:
        restpd[FAULT_FLAG] = ttopdownpd[FAULT_FLAG]
    restpd["preFlag"] = resList
    return restpd
