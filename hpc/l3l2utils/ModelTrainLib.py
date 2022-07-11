import os
from typing import Dict, List

import pandas as pd

from hpc.classifiers.ModelChange import change_threshold
from hpc.l3l2utils.DataFrameOperation import mergeDataFrames, mergeProceeDF, smoothseries, getSeriesFrequencyMeanLists, \
    getSeriesFrequencyMean, getSeriesMinFrequencyMeanLists, getSeriesMaxFrequencyMeanLists
from hpc.l3l2utils.DataFrameSaveRead import savepdfile
from hpc.l3l2utils.DataOperation import changeTimePdsToStrFromInt, changeTimeToFromPdlists, getsametimepdList
from hpc.l3l2utils.DefineData import usefulFeatures, TIME_COLUMN_NAME, FAULT_FLAG, MODEL_TYPE
from hpc.l3l2utils.FeatureExtraction import differenceProcess, differenceServer
from hpc.l3l2utils.ParsingJson import readJsonToDict, getServerPdFromJsonDict, getProcessPdFromJsonDict, \
    getL2PdFromJsonDict, getNetworkPdFromJsonDict, getPingPdFromJsonDict, getTopdownPdFromJsonDict

"""
2. memleakmin  内存泄露的最小值
3. pgfree_thread pgfree的变化幅度
4. ddrc_ddwr_sum_max的变化幅度
5. maxflopsinio mflops的变化幅度
"""

"""
函数功能：将modeltrain的config中的正常训练数据和异常训练数据取得
"""
def getNormalDectionJson(modeltrainconfig: Dict, normalDectionjson: Dict = None) -> Dict:
    if normalDectionjson is None and modeltrainconfig["trainNormalDataPath"] is not None:
        normalDectionjson=readJsonToDict(*(os.path.split(modeltrainconfig["trainNormalDataPath"])))
    return normalDectionjson
"""
函数功能：将modeltrain的config中的正常训练数据和异常训练数据取得
"""
def getTrainDectionJson(modeltrainconfig: Dict, abnormalDectionjson: Dict = None) -> Dict:
    if abnormalDectionjson is None:
        abnormalDectionjson=readJsonToDict(*(os.path.split(modeltrainconfig["trainAbnormalDataPath"])))
    return abnormalDectionjson
"""
函数功能：从DectionJson中获得需要的数据并且进行简单的时间和差分处理
"""
def getAllDataFramesFromDectionJson(dataDectionJson: Dict, isTimeisStr: bool=True)->Dict:
    if dataDectionJson is None:
        return None
    predictserverpds = getServerPdFromJsonDict(sdict=dataDectionJson)
    predictprocesspds = getProcessPdFromJsonDict(sdict=dataDectionJson)
    predictl2pds = getL2PdFromJsonDict(sdict=dataDectionJson)
    predictnetworkpds = getNetworkPdFromJsonDict(sdict=dataDectionJson)
    predictpingpds = getPingPdFromJsonDict(sdict=dataDectionJson)
    predicttopdwnpds = getTopdownPdFromJsonDict(sdict=dataDectionJson)
    if not isTimeisStr:
        changeTimePdsToStrFromInt(predictserverpds)
        changeTimePdsToStrFromInt(predictprocesspds)
        changeTimePdsToStrFromInt(predictl2pds)
        changeTimePdsToStrFromInt(predictnetworkpds)
        changeTimePdsToStrFromInt(predictpingpds)
        changeTimePdsToStrFromInt(predicttopdwnpds)
    # 时间统一处理
    predictserverpds = changeTimeToFromPdlists(predictserverpds, isremoveDuplicate=True)
    predictprocesspds = changeTimeToFromPdlists(predictprocesspds)
    predictl2pds = changeTimeToFromPdlists(predictl2pds, isremoveDuplicate=True)
    predictnetworkpds = changeTimeToFromPdlists(predictnetworkpds, isremoveDuplicate=True)
    predictpingpds = changeTimeToFromPdlists(predictpingpds, isremoveDuplicate=False)
    predicttopdwnpds = changeTimeToFromPdlists(predicttopdwnpds, isremoveDuplicate=True)

    # 差分处理
    # 对异常数据进行差分处理之后，得到cpu特征值 process数据会去掉前三min和后三min的数值
    predictprocesspds = differenceProcess(predictprocesspds, usefulFeatures["process_diff"])
    # 对异常server服务数据进行差分处理之后，得到一些指标
    predictserverpds = differenceServer(predictserverpds, usefulFeatures["server_diff"])
    predictnetworkpds = differenceServer(predictnetworkpds, usefulFeatures["network_diff"])
    predictl2pds = differenceServer(predictl2pds, usefulFeatures["compute_diff"])
    predictpingpds = differenceServer(predictpingpds, usefulFeatures["ping_diff"])
    predicttopdwnpds = differenceServer(predicttopdwnpds, usefulFeatures["topdown_diff"])
    resDict = {
        "server": mergeDataFrames(predictserverpds),
        "process": mergeDataFrames(predictprocesspds),
        "network": mergeDataFrames(predictnetworkpds),
        "compute": mergeDataFrames(predictl2pds),
        "ping": mergeDataFrames(predictpingpds),
        "topdown": mergeDataFrames(predicttopdwnpds)
    }
    return resDict

"""
函数功能：从表格中提取标签为abnormalType的异常值
"""
def abstractAbnormalData(datadf: pd.DataFrame, abnormalType: List[int], labelFlag: str=FAULT_FLAG):
    return datadf[datadf[labelFlag].isin(abnormalType)]
def abstractMinMean(datadf: pd.DataFrame,featurename: str, abnormalType: List[int], labelFlag: str=FAULT_FLAG):
    minValue = -1
    for iabtype in abnormalType:
        if iabtype not in datadf[FAULT_FLAG].tolist():
            continue
        tdf = abstractAbnormalData(datadf, [iabtype], labelFlag=labelFlag)
        tminvalues = getSeriesFrequencyMean(tdf[featurename])
        if minValue == -1:
            minValue = tminvalues
            continue
        minValue = min(minValue, tminvalues)
    return minValue

"""
得到内存泄露是内存的变化大小
"""
def getMemLeakPermin(normalfilepdDict: Dict, abnormalfilepdDict: Dict, modelconfigJson: Dict=None, )->float:
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
        respd["server_mem_used"] = servermem
        respd["process_mem_used"] = processmem
        respd["mem_sub"] = othermemdiff
        respd[FAULT_FLAG] = pspd[FAULT_FLAG]
        return respd
    # memleak从abnormalleak得到
    processpd = abnormalfilepdDict["process"].copy()
    serverpd = abnormalfilepdDict["server"].copy()
    memorypd = getMemory(serverpd=serverpd, processpd=processpd)
    # 取 61 62 63 64 65中最小值的变化值为判断依据, 最后减少100M作为
    memory60_mean = getSeriesMinFrequencyMeanLists(memorypd, labels=modelconfigJson["memleaklabels"], features=["mem_sub"])["mem_sub"]
    memory60_mean_1 = get_series_boundary_in_labels(memorypd, modelconfigJson['memleaklabels'], 'mem_sub', 'min')
    memory60_meanp80 = memory60_mean * 0.8
    memorypd["memsub60_mean"] = memory60_mean
    memorypd["memsub60_mean_1"] = memory60_mean_1
    memorypd["memsub60_mean_p80"] = memory60_meanp80
    if modelconfigJson["debugpath"] is not None:
        tpath = os.path.join(modelconfigJson["debugpath"], "memleak60")
        savepdfile(memorypd, tpath, "memleakdebug.csv")
    return memory60_mean_1


"""
得到mflops变化幅度的大小
"""
def getMaxflopsinio(normalfilepdDict: Dict, abnormalfilepdDict: Dict, modelconfigJson: Dict=None, )->int:
    normaltopdowndf = normalfilepdDict["topdown"].copy()
    abnormaltopdowndf = abnormalfilepdDict["topdown"].copy()
    # time normalmflops abnormalmflops normalmean abnormalmean abnormal50mean
    timeseries = normaltopdowndf[TIME_COLUMN_NAME]
    if len(abnormaltopdowndf) > len(normaltopdowndf):
        timeseries = abnormaltopdowndf[TIME_COLUMN_NAME]

    # 先进行mflops的平滑操作
    cname = "mflops"
    normaltopdowndf[cname] = normaltopdowndf[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    normaltopdowndf[cname] = normaltopdowndf[cname].rolling(window=5, center=True, min_periods=1).mean()
    abnormaltopdowndf[cname] = abnormaltopdowndf[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    abnormaltopdowndf[cname] = abnormaltopdowndf[cname].rolling(window=5, center=True, min_periods=1).mean()
    # 获得正常值情况下的mflops
    normalmflopsmean = getSeriesFrequencyMean(normaltopdowndf[cname])
    # 获得 异常情况下mflops的平均值
    abnormaltopdowndfmean = getSeriesFrequencyMean(abnormaltopdowndf[cname])
    # 获得50,90标签中的最小值
    abnormaltopdowndf5090mean = getSeriesMinFrequencyMeanLists(abnormaltopdowndf, labels=modelconfigJson["memorybandwidthlabels"] + modelconfigJson["cachegrablabels"], features=[cname])[cname]

    abnormaltopdowndf5090meanp90 = abnormaltopdowndf5090mean * 0.9
    if modelconfigJson["debugpath"] is not None:
        # 保存为debugpd
        serieslist = [
            timeseries,
            pd.Series(name="normal_mflops", data=normaltopdowndf[cname]),
            pd.Series(name="abnormal_mflops", data=abnormaltopdowndf[cname]),
            pd.Series(name=FAULT_FLAG, data=abnormaltopdowndf[FAULT_FLAG]),
        ]
        debugpd = pd.concat(serieslist, axis=1)
        debugpd["normal_mflops_mean"] = normalmflopsmean
        debugpd["abnormal_mflops_mean"] = abnormaltopdowndfmean
        debugpd["abnormal_mflops_mean_5090"] = abnormaltopdowndf5090mean
        debugpd["abnormaltopdowndf5090meanp90"] = abnormaltopdowndf5090meanp90
        debugpd.fillna(-1, inplace=True)
        tpath = os.path.join(modelconfigJson["debugpath"], "mflopsdebug")
        savepdfile(debugpd, tpath, "mflopsdebug.csv")
    return abnormaltopdowndf5090meanp90

"""
函数功能：补偿pgfree之后
"""
def getPgfreeThread(normalfilepdDict: Dict, abnormalfilepdDict: Dict,maxflopsinio: float ,modelconfigJson: Dict=None, debugDict: Dict=None)->int:
    # mflops_change normal_mflops_mean
    debugpd = pd.DataFrame()
    def getMflopschange(itopdownpd: pd.DataFrame, mflops_mean: float, maxflopsinio: float) -> pd.Series:
        cname = "mflops"
        # itopdownpd = removeUselessDataFromTopdownList([itopdownpd])[0]
        mflops_normal_iomax = maxflopsinio
        # 将小于iomax的mflops设置为平均值
        itopdownpd[cname] = itopdownpd[cname].apply(lambda x: mflops_mean if x < mflops_normal_iomax else x)
        itopdownpd[cname] = itopdownpd[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
        mflops_change = itopdownpd[cname].apply(lambda x: (mflops_mean - x) / mflops_mean if x < mflops_mean else 0)
        # 将较高的mflpos_change抹为0
        # mflops_change.apply(lambda x: if x > )
        itopdownpd["mflops_change"] = mflops_change
        return mflops_change
        # 保证iserverpds和itodownpds时间与时间相互匹配

    def compensatePgfree(iserverpd: pd.DataFrame, itopdowndpd: pd.DataFrame,pgfree_mean: float ,mflops_mean: float, maxflopsinio: float, inplace=True):
        assert len(iserverpd) == len(itopdowndpd)
        if inplace:
            iserverpd = iserverpd.copy()
            itopdowndpd = itopdowndpd.copy()
        # 对iprocess和servercpu中的
        mflops_change = getMflopschange(itopdownpd=itopdowndpd, mflops_mean=mflops_mean, maxflopsinio=maxflopsinio)
        changes = mflops_change
        debugpd["mflops_change"] = changes
        cname = "pgfree"
        iserverpd[cname] = iserverpd[cname] + pgfree_mean * changes
        iserverpd[cname] = iserverpd[cname].rolling(window=5, center=True,
                                                    min_periods=1).median()  # 对pgfree得到的结果重新去掉最大值最小值
        # pgfree 需要减去平均值
        return iserverpd
    # 读入需要的数据
    normaltopdowndf = normalfilepdDict["topdown"].copy()
    normalserverdf = normalfilepdDict["server"].copy()
    abnormaltopdowndf = abnormalfilepdDict["topdown"].copy()
    abnormalserverdf = abnormalfilepdDict["server"].copy()
    # 将异常中的server和topdown时间取交集
    abnormaltopdowndf, abnormalserverdf = getsametimepdList([abnormaltopdowndf, abnormalserverdf])
    normaltopdowndf, normalserverdf = getsametimepdList([normaltopdowndf, normalserverdf])
    debugpd[TIME_COLUMN_NAME] = abnormaltopdowndf[TIME_COLUMN_NAME]
    debugpd[FAULT_FLAG] = abnormaltopdowndf[FAULT_FLAG]
    #
    # 得到normal中的mflops正常指标
    cname = "mflops"
    normaltopdowndf[cname] = normaltopdowndf[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    normaltopdowndf[cname] = normaltopdowndf[cname].rolling(window=5, center=True, min_periods=1).mean()
    normalmflopsmean = getSeriesFrequencyMean(normaltopdowndf[cname])
    # 存储mflops的平均值
    debugDict["normalDataMean"]["topdown"]["mflops"] = normalmflopsmean
    debugDict["normalDataMean"]["topdown"]["pg_mflops"] = normalmflopsmean

    debugpd["normal_mflops_mean"] = normalmflopsmean
    cname = "pgfree"
    normalserverdf[cname] = normalserverdf[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    normalserverdf[cname] = normalserverdf[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    normalserverdf[cname] = normalserverdf[cname].rolling(window=5, center=True, min_periods=1).mean()
    normalpgfreemean = getSeriesFrequencyMean(normalserverdf[cname])
    debugpd["normal_pgfree_mean"] = normalpgfreemean
    # 存储mflops的平均值
    debugDict["normalDataMean"]["server"]["pgfree"] = normalpgfreemean

    cname = "mflops"
    abnormaltopdowndf[cname] = abnormaltopdowndf[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    abnormaltopdowndf[cname] = abnormaltopdowndf[cname].rolling(window=5, center=True, min_periods=1).mean()
    abnormalmflopsmean = getSeriesFrequencyMean(abnormaltopdowndf[cname])
    debugpd["abnormal_mflop_mean"] = abnormalmflopsmean
    cname = "pgfree"
    abnormalserverdf[cname] = abnormalserverdf[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    abnormalserverdf[cname] = abnormalserverdf[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    abnormalserverdf[cname] = abnormalserverdf[cname].rolling(window=5, center=True, min_periods=1).mean()
    debugpd["abnoraml_pgfree_origin"] = abnormalserverdf[cname]
    abnormalpgfreemean = getSeriesFrequencyMean(abnormalserverdf[cname])
    debugpd["abnoraml_allpgfree_mean"] = abnormalpgfreemean
    # 得到补偿得到pgfree
    abnormalcpgfreedf = compensatePgfree(abnormalserverdf,abnormaltopdowndf,normalpgfreemean, normalmflopsmean, maxflopsinio)
    debugpd["abnoraml_pgfree_compensation"] = abnormalcpgfreedf["pgfree"]
    # 提取52 53 54 55的最小值
    # pgfree_abnorma50_mean = abstractMinMean(abnormalcpgfreedf, "pgfree", [52, 53, 54, 55])
    pgfree_abnorma50_mean =  getSeriesMinFrequencyMeanLists(abnormalcpgfreedf, labels=modelconfigJson["memorybandwidthlabels"], features=["pgfree"])["pgfree"]
    debugpd["abnormal_abnormalpgfree_mean"] = pgfree_abnorma50_mean

    pgfreescope = (pgfree_abnorma50_mean - normalpgfreemean) * 0.9
    debugpd["abnormal_abnormalpgfree_addmean"] = normalpgfreemean + pgfreescope

    if modelconfigJson["debugpath"] is not None:
        debugpd = pd.concat([debugpd, pd.Series(name="normal_pgfree", data=normalserverdf["pgfree"])], axis=1)
        debugpd.fillna(-1, inplace=True)
        tpath = os.path.join(modelconfigJson["debugpath"], "pgfree_thread")
        savepdfile(debugpd, tpath, "pgfree_thread.csv")
    return pgfreescope

"""
得到ddrc_ddwr_sum变化的幅度
"""
def getddrc_ddwr_sumscope(normalfilepdDict: Dict, abnormalfilepdDict: Dict,maxflopsinio: float ,modelconfigJson: Dict=None, debugDict: Dict=None)->int:
    debugpd = pd.DataFrame()
    def getMflopschange(itopdownpd: pd.DataFrame, mflops_mean: float, maxflopsinio: float) -> pd.Series:
        cname = "mflops"
        # itopdownpd = removeUselessDataFromTopdownList([itopdownpd])[0]
        mflops_normal_iomax = maxflopsinio
        # 将小于iomax的mflops设置为平均值
        itopdownpd[cname] = itopdownpd[cname].apply(lambda x: mflops_mean if x < mflops_normal_iomax else x)
        itopdownpd[cname] = itopdownpd[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
        mflops_change = itopdownpd[cname].apply(lambda x: (mflops_mean - x) / mflops_mean if x < mflops_mean else 0)
        # 将较高的mflpos_change抹为0
        # mflops_change.apply(lambda x: if x > )
        itopdownpd["mflops_change"] = mflops_change
        return mflops_change
        # 保证iserverpds和itodownpds时间与时间相互匹配
    # 读入需要的topdown数据
    normaltopdowndf = normalfilepdDict["topdown"].copy()
    abnormaltopdowndf = abnormalfilepdDict["topdown"].copy()

    # 存储时间
    debugpd[TIME_COLUMN_NAME] = abnormaltopdowndf[TIME_COLUMN_NAME]
    debugpd[FAULT_FLAG] = abnormaltopdowndf[FAULT_FLAG]


    # 得到mflops的正常稳定值
    cname = "mflops"
    normaltopdowndf[cname] = normaltopdowndf[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    normaltopdowndf[cname] = normaltopdowndf[cname].rolling(window=5, center=True, min_periods=1).mean()
    abnormaltopdowndf[cname] = abnormaltopdowndf[cname].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    abnormaltopdowndf[cname] = abnormaltopdowndf[cname].rolling(window=5, center=True, min_periods=1).mean()
    normalmflopsmean = getSeriesFrequencyMean(normaltopdowndf[cname])
    abnormalmflopsmean = getSeriesFrequencyMean(abnormaltopdowndf[cname])
    # 存储mflops的平均值
    debugDict["normalDataMean"]["topdown"]["d_mflops"] = normalmflopsmean

    debugpd["normal_mlops_mean"] = normalmflopsmean
    debugpd["abnormal_allmflops_mean"] = abnormalmflopsmean
    rd_name = "ddrc_rd"
    normaltopdowndf[rd_name] = normaltopdowndf[rd_name].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    normaltopdowndf[rd_name] = normaltopdowndf[rd_name].rolling(window=5, center=True, min_periods=1).mean()
    abnormaltopdowndf[rd_name] = abnormaltopdowndf[rd_name].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    abnormaltopdowndf[rd_name] = abnormaltopdowndf[rd_name].rolling(window=5, center=True, min_periods=1).mean()
    normalddrdmean = getSeriesFrequencyMean(normaltopdowndf[rd_name])
    abnormalddrdmean = getSeriesFrequencyMean(abnormaltopdowndf[rd_name])
    debugpd["normalddrdmean"] = normalddrdmean
    debugpd["abnormalddrdmean"] = abnormalddrdmean
    # 存储ddrc_rd的平均值
    debugDict["normalDataMean"]["topdown"]["ddrc_rd"] = normalddrdmean

    wr_name = "ddrc_wr"
    normaltopdowndf[wr_name] = normaltopdowndf[wr_name].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    normaltopdowndf[wr_name] = normaltopdowndf[wr_name].rolling(window=5, center=True, min_periods=1).mean()
    abnormaltopdowndf[wr_name] = abnormaltopdowndf[wr_name].rolling(window=5, center=True, min_periods=1).median()  # 先将最大最小值去除
    abnormaltopdowndf[wr_name] = abnormaltopdowndf[wr_name].rolling(window=5, center=True, min_periods=1).mean()
    normalddwrmean = getSeriesFrequencyMean(normaltopdowndf[wr_name])
    abnormalddwrmean = getSeriesFrequencyMean(abnormaltopdowndf[wr_name])
    debugpd["normalddwrmean"] = normalddwrmean
    debugpd["abnormalddwrmean"] = abnormalddwrmean
    # 存储ddrc_wr的平均值
    debugDict["normalDataMean"]["topdown"]["ddrc_wr"] = normalddwrmean

    # 计算mflops_changes
    mflop_change = getMflopschange(abnormaltopdowndf, normalmflopsmean, maxflopsinio)
    debugpd["mflops_change"] = mflop_change
    # 计算补偿之后的值

    debugpd["ddrc_rd"] = abnormaltopdowndf["ddrc_rd"]
    debugpd["ddrc_wr"] = abnormaltopdowndf["ddrc_wr"]
    abnormaltopdowndf["ddrc_rd"] = abnormaltopdowndf["ddrc_rd"] + normalddrdmean * mflop_change
    abnormaltopdowndf["ddrc_wr"] = abnormaltopdowndf["ddrc_wr"] + normalddwrmean * mflop_change
    debugpd["ddrc_rd_compensation"] = abnormaltopdowndf["ddrc_rd"]
    debugpd["ddrc_wr_compensation"] = abnormaltopdowndf["ddrc_wr"]

    # 求解rd_wr_sum
    rd_wr_cname = "ddrc_ddwr_sum"
    normaltopdowndf[rd_wr_cname] = normaltopdowndf[rd_name] + normaltopdowndf[wr_name]
    normaltopdowndf[rd_wr_cname] = normaltopdowndf[rd_wr_cname].rolling(window=5, center=True, min_periods=1).median()
    abnormaltopdowndf[rd_wr_cname] = abnormaltopdowndf[rd_name] + abnormaltopdowndf[wr_name]
    abnormaltopdowndf[rd_wr_cname] = abnormaltopdowndf[rd_wr_cname].rolling(window=5, center=True, min_periods=1).median()
    debugpd["ddrc_ddwr_sum"] = abnormaltopdowndf[rd_wr_cname]

    normal_rd_wr_mean = normalddrdmean + normalddwrmean
    # 存储ddrc_rd的平均值
    debugDict["normalDataMean"]["topdown"]["ddrc_ddwr_sum"] = normal_rd_wr_mean

    debugpd["normal_ddrc_ddwr_sum_mean"] = normal_rd_wr_mean
    abnoraml_rd_wr_mean = abnormalddrdmean + abnormalddwrmean
    debugpd["abnormal_ddrc_ddwr_sum_mean"] = abnoraml_rd_wr_mean
    # 求解51 52 53 54 55 91 92 93 94 95
    # abnoraml_rd_wr_mean5090 = abstractMinMean(abnormaltopdowndf, rd_wr_cname, [51,52,53,54,55,91,92,93,94,95])
    abnoraml_rd_wr_mean5090 = getSeriesMinFrequencyMeanLists(abnormaltopdowndf, labels=modelconfigJson["memorybandwidthlabels"] + modelconfigJson["cachegrablabels"], features=["ddrc_ddwr_sum"])["ddrc_ddwr_sum"]
    debugpd["ddrc_ddwr_sum_abnormal5090mean"] = abnoraml_rd_wr_mean5090
    rd_wr_scope = (abnoraml_rd_wr_mean5090 - normal_rd_wr_mean) * 0.9
    debugpd["ddrc_ddwr_sum_abnormal5090meanaddp90"] = normal_rd_wr_mean + rd_wr_scope
    if modelconfigJson["debugpath"] is not None:
        pdlists = [
            debugpd,
            pd.Series(name="normal_ddrd", data=normaltopdowndf[rd_name]),
            pd.Series(name="normal_ddwr", data=normaltopdowndf[wr_name]),
            pd.Series(name="normal_ddrc_ddwr_sum", data=normaltopdowndf[rd_wr_cname]),
        ]
        debugpd = pd.concat(pdlists, axis=1)
        debugpd.fillna(-1, inplace=True)
        tpath = os.path.join(modelconfigJson["debugpath"], "ddrc_ddwr_sum")
        savepdfile(debugpd, tpath, "ddrc_ddwr_sum.csv")
    return rd_wr_scope

def changeModel(configJsonDict: Dict, outputJsonDict: Dict):
    # 修改内存泄漏模型
    memleakmin = outputJsonDict["memleakpermin"]
    memleakpath = os.path.join(configJsonDict["servermemory_modelpath"], MODEL_TYPE[0] + ".pkl")
    change_threshold(memleakpath, 0, memleakmin)

    # 修改内存带宽模型
    pgfree_thread = outputJsonDict["pgfree_thread"]
    memorybandwidthpath = os.path.join(configJsonDict["serverbandwidth_modelpath"], MODEL_TYPE[0] + ".pkl")
    change_threshold(memorybandwidthpath, 0, pgfree_thread)
    # 修改cache抢占模型
    ddrc_ddwr_sum_max = outputJsonDict["ddrc_ddwr_sum_max"]
    cachegrab_modelpath = os.path.join(configJsonDict["cachegrab_modelpath"], MODEL_TYPE[0] + ".pkl")
    change_threshold(cachegrab_modelpath, 0, ddrc_ddwr_sum_max)

"""
根据全CPU得到CPU下降的时间
"""
def getCPUTimeThread(normalfilepdDict: Dict, abnormalfilepdDict: Dict, modelconfigJson: Dict=None, debugDict: Dict=None) -> float:
    def getpidcpuInfo(processpd: pd.DataFrame):
        respd = pd.DataFrame(index=processpd["time"].drop_duplicates())
        processpd.set_index("time", inplace=True)
        for icore, icorepd in processpd.groupby("cpu_affinity"):
            # icorepd = icorepd.reset_index(drop=True)
            # 去重
            icorepd = icorepd[~icorepd.index.duplicated(keep="first")]
            cname = "core{}_cpu".format(icore)
            cpuSeries = pd.Series(data=icorepd["cpu"], name=cname)
            respd = pd.concat([respd, cpuSeries], axis=1)
            if FAULT_FLAG not in respd.columns.tolist():
                respd[FAULT_FLAG] = icorepd[FAULT_FLAG]
        respd.fillna(-1, inplace=True)
        respd.index.name = "time"
        respd.reset_index(drop=False, inplace=True)
        return respd

    normalprocessdf = normalfilepdDict["process"].copy()
    abnormalprocessdf = abnormalfilepdDict["process"].copy()
    debugpd = getpidcpuInfo(abnormalprocessdf)

    # 将abnormalprocessdf变成第0个核心为主的 保证第0个核心是有所注入的
    abnormalprocessdf = abnormalprocessdf[abnormalprocessdf["cpu_affinity"] == 0]

    # 得到CPU异常类型的最小变化
    cpunormalmean = getSeriesFrequencyMeanLists(normalprocessdf, ["cpu"])["cpu"]
    # 存储cpunormalmean的平均值
    debugDict["normalDataMean"]["server"]["cpu"] = cpunormalmean

    cpuabnormalmean = getSeriesMaxFrequencyMeanLists(abnormalprocessdf, labels=modelconfigJson["allcpulabels"], features=["cpu"])["cpu"]
    cpuabnormalmean_use = cpunormalmean - (cpunormalmean - cpuabnormalmean) * 0.9
    debugpd["cpunormalmean"] = cpunormalmean
    debugpd["cpuabnormalmean"] = cpuabnormalmean
    debugpd["cpuabnormalmean_use"] = cpuabnormalmean_use

    if modelconfigJson["debugpath"] is not None:
        tpath = os.path.join(modelconfigJson["debugpath"], "abnormalCpuTimeThread")
        savepdfile(debugpd, tpath, "abnormalCpuTimeThread.csv")
    return cpuabnormalmean_use

"""
通过load1来判断
"""
def getRandomcpuThreshold(normalfilepdDict: Dict, abnormalfilepdDict: Dict, modelconfigJson: Dict=None, debugDict: Dict=None) -> float:
    normalserverdf = normalfilepdDict["server"].copy()
    abnormalserverdf = abnormalfilepdDict["server"].copy()
    debugpd = pd.DataFrame()

    # 存储时间
    debugpd[TIME_COLUMN_NAME] = abnormalserverdf[TIME_COLUMN_NAME]
    debugpd[FAULT_FLAG] = abnormalserverdf[FAULT_FLAG]
    debugpd["load1"] = abnormalserverdf["load1"]
    # 正常load1
    normalload1mean = getSeriesFrequencyMeanLists(normalserverdf, ["load1"])["load1"]
    # 存储load1的平均值
    debugDict["normalDataMean"]["server"]["load1"] = normalload1mean

    abnormalload1mean = getSeriesFrequencyMeanLists(abnormalserverdf, ["load1"])["load1"]
    alllabels = modelconfigJson["randomcpulabels"] + modelconfigJson["memorybandwidthlabels"] + modelconfigJson["cachegrablabels"]
    abnormalload1mean508090 = getSeriesMinFrequencyMeanLists(abnormalserverdf, labels=alllabels, features=["load1"])["load1"]
    abnormalload1mean508090_1 = get_series_boundary_in_labels(abnormalserverdf, alllabels, 'load1', 'min')

    debugpd["nomalload1mean"] = normalload1mean
    debugpd["abnormalddrdmean"] = abnormalload1mean
    debugpd["abnormalload1mean508090"] = abnormalload1mean508090
    debugpd["abnormalload1mean508090_1"] = abnormalload1mean508090_1
    debugpd["abnormalload1mean508090p90"] = normalload1mean + (abnormalload1mean508090 - normalload1mean) * 0.9
    if modelconfigJson["debugpath"] is not None:
        tpath = os.path.join(modelconfigJson["debugpath"], "randomcpuThreshold")
        savepdfile(debugpd, tpath, "randomcpuThreshold.csv")
    return (abnormalload1mean508090_1 - normalload1mean) * 0.9
    # return (abnormalload1mean508090 - normalload1mean) * 0.9


"""  
必须保证freq
返回111 121 131 141 151 161 这几个里面的freq
针对L2异常导致的频率freq下降，判断下降的比例
如果无法判断，那么返回None
"""
def getFreqDownThresholdpercent(normalfilepdDict: Dict, abnormalfilepdDict: Dict, modelconfigJson: Dict=None, debugDict: Dict=None) -> float:
    normalserverdf = normalfilepdDict["server"].copy()
    abnormalserverdf = abnormalfilepdDict["server"].copy()

    # 存储时间
    debugpd = pd.DataFrame()
    debugpd[TIME_COLUMN_NAME] = abnormalserverdf[TIME_COLUMN_NAME]
    debugpd[FAULT_FLAG] = abnormalserverdf[FAULT_FLAG]
    debugpd["freq"] = abnormalserverdf["freq"]

    # 平滑freq
    cname = "freq"
    normalserverdf[cname] = smoothseries(normalserverdf[cname])
    abnormalserverdf[cname] = smoothseries(abnormalserverdf[cname])

    normalfreqmean = getSeriesFrequencyMeanLists(normalserverdf, [cname])[cname]
    # 存储freq的平均值
    debugDict["normalDataMean"]["compute"]["freq"] = normalfreqmean

    abnormalfreqmean = getSeriesFrequencyMeanLists(abnormalserverdf, [cname])[cname]
    abnormal_abfreqmean = getSeriesMaxFrequencyMeanLists(abnormalserverdf, labels=modelconfigJson["l2labels"], features=["freq"])["freq"]

    debugpd["normalfreqmean"] = normalfreqmean
    debugpd["abnormalfreqmean"] = abnormalfreqmean
    debugpd["abnormal_abfreqmean"] = abnormal_abfreqmean

    if modelconfigJson["debugpath"] is not None:
        tpath = os.path.join(modelconfigJson["debugpath"], "freqDownThresholdpercent")
        savepdfile(debugpd, tpath, "freqDownThresholdpercent.csv")

    resvalue = (normalfreqmean - abnormal_abfreqmean) / normalfreqmean * 100 * 0.9
    return resvalue
"""
必须有121的存在
通过121 得到power的变化情况
"""
def getPowerThreshold(normalfilepdDict: Dict, abnormalfilepdDict: Dict, modelconfigJson: Dict=None, debugDict: Dict=None) -> float:
    normalcomputedf = normalfilepdDict["compute"].copy()
    abnormalcomputedf = abnormalfilepdDict["compute"].copy()
    # 存储时间
    debugpd = pd.DataFrame()
    debugpd[TIME_COLUMN_NAME] = abnormalcomputedf[TIME_COLUMN_NAME]
    debugpd[FAULT_FLAG] = abnormalcomputedf[FAULT_FLAG]
    debugpd["power"] = abnormalcomputedf["power"]
    debugpd["cpu_power"] = abnormalcomputedf["cpu_power"]

    cname = "power"
    normalcomputedf[cname] = smoothseries(normalcomputedf[cname])
    abnormalcomputedf[cname] = smoothseries(abnormalcomputedf[cname])

    normalpowermean = getSeriesFrequencyMeanLists(normalcomputedf, [cname])[cname]
    # 存储power的平均值
    debugDict["normalDataMean"]["compute"]["power"] = normalpowermean

    abnormalpowermean = getSeriesFrequencyMeanLists(abnormalcomputedf, [cname])[cname]
    abnormal_abpowermean = getSeriesMaxFrequencyMeanLists(abnormalcomputedf, labels=modelconfigJson["l2labels"], features=["power"])["power"]
    debugpd["normalpowermean"] = normalpowermean
    debugpd["abnormalpowermean"] = abnormalpowermean
    debugpd["abnormal_abpowermean"] = abnormal_abpowermean
    if modelconfigJson["debugpath"] is not None:
        tpath = os.path.join(modelconfigJson["debugpath"], "power_threshold")
        savepdfile(debugpd, tpath, "power_threshold.csv")
    # resvalue = abnormal_abpowermean / normalpowermean * 100 * 1.1
    resvalue = 100 - (normalpowermean - abnormal_abpowermean) / normalpowermean * 100 * 0.9
    return resvalue

# jinxing write

def get_series_boundary(data_series: pd.Series, min_or_max: str):
    from collections import Counter
    from sklearn.mixture import BayesianGaussianMixture
    data = pd.DataFrame(data_series)
    # tags = BayesianGaussianMixture(n_components=3, random_state=0).fit_predict(data)
    # most_tag = Counter(tags).most_common(1)[0][0]
    from sklearn.ensemble import IsolationForest
    tags = IsolationForest(random_state=0).fit_predict(data)
    most_tag = 1
    if min_or_max == 'min':
        boundary = data.iloc[tags == most_tag].min()
    else:
        boundary = data.iloc[tags == most_tag].max()
    return float(boundary)


def get_series_boundary_in_labels(data: pd.DataFrame, labels: List[int], feature: str, min_or_max: str):
    boundaries = []
    # 计算每个标签中feature的边界值
    for label in labels:
        if label not in data[FAULT_FLAG].tolist():
            continue
        label_data = data[data[FAULT_FLAG] == label]
        boundaries.append(get_series_boundary(label_data[feature], min_or_max))
    # 计算所有边界值中的最小/大值
    if min_or_max == 'min':
        boundary = min(boundaries)
    else:
        boundary = max(boundaries)
    return boundary









