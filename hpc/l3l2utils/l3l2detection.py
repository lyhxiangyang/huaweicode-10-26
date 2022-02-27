import os.path
from itertools import chain
from typing import List, Tuple, Dict, Set

import pandas as pd

from hpc.l3l2utils.DataFrameSaveRead import savepdfile
from hpc.l3l2utils.DataOperation import remove_Abnormal_Head_Tail, removeAllHeadTail
from hpc.l3l2utils.DefineData import FAULT_FLAG, TIME_COLUMN_NAME, errorFeatureDict, CPU_ABNORMAL_TYPE, MEMORY_ABNORMAL_TYPE

"""
修改preFlag那些单独存在的点
"""


def fixIsolatedPointPreFlag(l2l3predetectresultpd: pd.DataFrame):
    l2l3predetectresultpd = l2l3predetectresultpd.copy()

    # 判断这个错误是否符合equalFlag
    # 如果exultFlags是奇数如11011那么ipos指向的是0中间位置，如果exultFlags是偶数，如11100111那么ipos指向的是第一个0位置
    # 如果是11111000011111 中间位置的第1个位置
    # 2 代表0或者1
    # iequelpos指要与ipos对齐的位置 从0开始记数
    def isallEqual(preFlagsList, equalFlags: List, ipos, iequalpos,
                   ifault):  # 比较两个列表是否相等  abnormals是preFlag， equalFlags必须是奇数, ipos是当前的位置
        beginpos = ipos - iequalpos
        endpos = ipos + len(equalFlags) - iequalpos

        abnormals = []
        for i in range(beginpos, endpos):
            if i < 0:
                abnormals.append([0])
            elif i >= len(preFlagsList):
                abnormals.append([0])
            else:
                abnormals.append(preFlagsList[i])

        if len(abnormals) != len(equalFlags):
            return False
        abnormals = [1 if ifault in ifautltlists else 0 for ifautltlists in abnormals]
        compareFlags = [abnormals[i] == equalFlags[i] if equalFlags[i] != 2 else True for i in range(0, len(abnormals))]  # True代表相同 False代表不同
        return compareFlags.count(True) == len(compareFlags)

    preflagList = list(l2l3predetectresultpd["preFlag"])
    allPreFlags = sorted(list(set(chain.from_iterable(preflagList))))  # 得到所有的错误
    if 0 in allPreFlags:
        allPreFlags.remove(0)

    # 为preflagList 从beginpos开始赋值value  长度为lenght, 此时的错误是ifault
    # isadd True添加    False 删除
    def assignmentValue(preflagList, beginpos, length, value):
        for i in range(beginpos, min(len(preflagList), beginpos + length)):
            if value not in preflagList[i]:
                preflagList[i].append(value)
    # 删除
    def removeValue(preflagList, beginpos, length, value):
        for i in range(beginpos, min(len(preflagList), beginpos + length)):
            if value in preflagList[i]:
                preflagList[i].remove(value)


    for ifault in allPreFlags:
        i = 0
        while i < len(preflagList):
            # 1.
            eintlist = list(map(int, list("00100")))
            lenerror = 1
            iequalpos = 2
            if isallEqual(preflagList, eintlist, i, iequalpos, ifault):
                if (eintlist[0] == 0):
                    removeValue(preflagList, i , lenerror, ifault)
                    assignmentValue(preflagList, i, lenerror, 0)
                elif (eintlist[0] == 1):
                    removeValue(preflagList, i , lenerror, 0)
                    assignmentValue(preflagList, i, lenerror, ifault)
                i += len(eintlist) - iequalpos
                continue
            # 2.
            eintlist = list(map(int, list("11011")))
            lenerror = 1
            iequalpos = 2
            if isallEqual(preflagList, eintlist, i, iequalpos, ifault):
                if (eintlist[0] == 0):
                    removeValue(preflagList, i , lenerror, ifault)
                    assignmentValue(preflagList, i, lenerror, 0)
                elif (eintlist[0] == 1):
                    removeValue(preflagList, i , lenerror, 0)
                    assignmentValue(preflagList, i, lenerror, ifault)
                i += len(eintlist) - iequalpos
                continue
            # 3.
            eintlist = list(map(int, list("11122211")))
            lenerror = 3
            iequalpos = 3
            isadd = eintlist[0] != 0 # 第1个是0 就删除，第1个是1就添加
            if isallEqual(preflagList, eintlist, i, iequalpos, ifault):
                if (eintlist[0] == 0):
                    removeValue(preflagList, i , lenerror, ifault)
                    assignmentValue(preflagList, i, lenerror, 0)
                elif (eintlist[0] == 1):
                    removeValue(preflagList, i , lenerror, 0)
                    assignmentValue(preflagList, i, lenerror, ifault)
                i += len(eintlist) - iequalpos
                continue
            # 4.
            eintlist = list(map(int, list("0002222000")))
            lenerror = 4
            iequalpos = 3
            if isallEqual(preflagList, eintlist, i, iequalpos, ifault):
                if (eintlist[0] == 0):
                    removeValue(preflagList, i , lenerror, ifault)
                    assignmentValue(preflagList, i, lenerror, 0)
                elif (eintlist[0] == 1):
                    removeValue(preflagList, i , lenerror, 0)
                    assignmentValue(preflagList, i, lenerror, ifault)
                i += len(eintlist) - iequalpos
                continue
            # 5.
            eintlist = list(map(int, list("01000")))
            lenerror = 1
            iequalpos = 1
            if isallEqual(preflagList, eintlist, i, iequalpos, ifault):
                if (eintlist[0] == 0):
                    removeValue(preflagList, i , lenerror, ifault)
                    assignmentValue(preflagList, i, lenerror, 0)
                elif (eintlist[0] == 1):
                    removeValue(preflagList, i , lenerror, 0)
                    assignmentValue(preflagList, i, lenerror, ifault)
                i += len(eintlist) - iequalpos
                continue

            i += 1


    # for i in range(0, len(preflagList)):
    #     # 对每个异常进行判断
    #     for ifault in allPreFlags:  # 对所有出现过的错误进行预判
    #         # 0 代表不是这个错误， 1代表是这个错误
    #         if isallEqual(preflagList, list(map(int, list("00100"))), i, ifault):
    #             preflagList[i].remove(ifault)
    #             continue  # 不需要这个异常
    #         if isallEqual(preflagList, list(map(int, list("11011"))), i, ifault):
    #             preflagList[i].append(ifault)
    #             continue
    #         if isallEqual(preflagList, list(map(int, list("1110111"))), i, ifault):
    #             preflagList[i].append(ifault)
    #             continue
    #         if isallEqual(preflagList, list(map(int, list("0001000"))), i, ifault):
    #             preflagList[i].append(ifault)
    #             continue
    #         if isallEqual(preflagList, list(map(int, list("0001000"))), i, ifault):
    #             preflagList[i].append(ifault)
    #             continue


    for i in range(0, len(preflagList)):
        # preflagList[i] = sorted(list(set(preflagList[i]))) # 可能有多个0
        preflagList[i].sort()
        if len(preflagList[i]) == 0:  # 全部删除干净了，那就等于0
            preflagList[i] = [0]
        if len(preflagList[i]) >= 2 and 0 in preflagList[i]:
            preflagList[i].remove(0)

    l2l3predetectresultpd["preFlag"] = preflagList
    l2l3predetectresultpd.reset_index(drop=True, inplace=True)
    return l2l3predetectresultpd


"""
对faultFlag进行修改
主要是将133变成131  134变成132
99 状态代表着这段时间没有wrf运行，可以删除
此外还要删除41-45  71-75  91-95
"""


def fixFaultFlag(l2l3predetectresultpd: pd.DataFrame):
    l2l3predetectresultpd = l2l3predetectresultpd.copy()
    l2l3predetectresultpd = remove_Abnormal_Head_Tail(l2l3predetectresultpd,
                                                      abnormals={41, 42, 43, 44, 45, 71, 72, 73, 74, 75, 91, 92, 51, 52}, windowsize=4)
    # 修改标签
    l2l3predetectresultpd.loc[:, FAULT_FLAG] = l2l3predetectresultpd.loc[:, FAULT_FLAG].apply(
        lambda x: 131 if x == 133 else x)
    l2l3predetectresultpd.loc[:, FAULT_FLAG] = l2l3predetectresultpd.loc[:, FAULT_FLAG].apply(
        lambda x: 132 if x == 134 else x)
    l2l3predetectresultpd.reset_index(drop=True, inplace=True)
    return l2l3predetectresultpd


"""
函数功能： 得到每个时间点的概率
"""


def getDetectionProbability(preFlagsList: List[List[int]]):
    def getBeforDataWeight(preFlags: List[List[int]], ipos: int, ifault: int, judgelen: int) -> List[float]:
        res = []
        for bpos in range(ipos - judgelen, ipos, 1):
            if bpos < 0:
                res.append(0)
                continue
            if ifault in preFlags[bpos]:
                res.append(1)
                continue
            res.append(0)
        return res

    def getAfterDataWeight(preFlags: List[List], ipos: int, ifault: int, judgelen: int) -> List[float]:
        res = []
        for bpos in range(ipos + 1, ipos + judgelen + 1, 1):
            if bpos >= len(preFlags):
                res.append(0)
                continue
            if ifault in preFlags[bpos]:
                res.append(1)
                continue
            res.append(0)
        return res

    def getp(weights: List[float], probabilities: List[float]) -> float:
        assert len(weights) == len(probabilities)
        return sum(weights[i] * probabilities[i] for i in range(0, len(weights)))

    # 得到时间概率的逻辑部分
    probabilityList = []
    preprobability = [0.1, 0.2, 0.3, 0.4]
    reverseprobability = list(reversed(preprobability))  # 概率相反
    judgeLen = len(preprobability)
    for ipos, onepreflag in enumerate(preFlagsList):
        oneprobabilityDict = {}
        for iflag in onepreflag:
            beforeWeight = getBeforDataWeight(preFlagsList, ipos, iflag, judgeLen)
            afterWeight = getAfterDataWeight(preFlagsList, ipos, iflag, judgeLen)
            nowprobability = 0.1 + 0.45 * (getp(beforeWeight, preprobability)) + 0.45 * (
                getp(afterWeight, reverseprobability))
            oneprobabilityDict[iflag] = nowprobability
        probabilityList.append(oneprobabilityDict)
    return probabilityList


"""
函数功能：根据预测信息得到一段时间内的每个时间段信息及其概率
"""


def getTimePeriodInfo(l2l3predetectresultpd: pd.DataFrame, preflagName: str = "preFlag",
                      probabilityName: str = "probability") -> List:
    # ========================================================================================================== 函数部分
    # 找到一个列表中连续不为0的数据的位置, 返回的是每段不为0的起始位置[4,9), 左闭右开
    def findAbnormalPos(flags: List[int]) -> List[Tuple[int, int]]:
        beginpos_endpos_List = []
        i = 0
        flagslen = len(flags)
        while i < flagslen:
            if flags[i] == [0]:
                i += 1
                continue
            beginpos = i
            while i < flagslen and flags[i] != [0]:
                i += 1
            endpos = i
            beginpos_endpos_List.append((beginpos, endpos))
        return beginpos_endpos_List

    # 根据位置的起始位置得到DataFrame
    def getDataFramesFromPos(pd: pd.DataFrame, pos: List[Tuple[int, int]]) -> List[pd.DataFrame]:
        pd = pd.copy()
        respdList = []
        for i in pos:
            # loc和使用[]是不一样的
            respdList.append(pd[i[0]:i[1]])
        return respdList

    # 计算一段时间的概率 ifault的概率
    def getProbability(prepd: pd.DataFrame, ifault: int, probabilityName: str = "probability") -> float:
        faultprob = 0
        prelen = len(prepd)
        probabilityLists = prepd[probabilityName]
        for iprobability in probabilityLists:
            iprobability: Dict
            if ifault in iprobability:
                faultprob += iprobability[ifault] / prelen
        return faultprob

    """
    得到预测值的错误类型 返回List， 概率从大到小排序
    返回一个List
    """

    def getErrorInfo(prepd: pd.DataFrame, preflagName: str = "preFlag", probabilityName: str = "probability") -> List:
        errorList = []
        preFlags = prepd[preflagName]
        allPreFlags = sorted(list(set(chain.from_iterable(preFlags))))
        for ifault in allPreFlags:
            errorDict = {}
            errorDict["type"] = ifault
            # 计算概率
            errorDict["probability"] = getProbability(prepd, ifault, probabilityName)
            errorList.append(errorDict)
        # 需要对errorList按照概率从大到小进行排序
        errorList = sorted(errorList, key=lambda x: x["probability"], reverse=True)
        return errorList

    """
    得到预测值所依赖的特征值及其重要程度
    将预测的这段时间内每个错误对应的特征值使用的特征百分比作为参考依据
    返回一个Dict
    """

    def getKPIInfo(prepd: pd.DataFrame, preflagName: str = "preFlag"):
        preFlags = prepd[preflagName]
        repeatPreFlags = list(chain.from_iterable(preFlags))
        alluserfeatureList = []
        for ifault in repeatPreFlags:
            if ifault not in errorFeatureDict:
                print("{} 的特征值没有存储".format(ifault))
                continue
            usedfeatures = errorFeatureDict[ifault]
            alluserfeatureList.extend(usedfeatures)

        # 得到所有错误使用的featurename，然后判断其占比进行输出
        alluserfeatureSeries = pd.Series(data=alluserfeatureList)
        proportionDict = dict(alluserfeatureSeries.value_counts(normalize=True, ascending=False))
        return proportionDict

    """
    返回影响程序的总体指标
    返回值float
    """

    def getperformance_var(prepd: pd.DataFrame, preflagName: str = "preFlag",
                           probabilityName: str = "probability") -> float:
        prepdlen = len(prepd)
        return -(1 - prepdlen / 10)

    if FAULT_FLAG in l2l3predetectresultpd.columns:
        reallabels = list(l2l3predetectresultpd[FAULT_FLAG])
        beginpos_endpos_list = findAbnormalPos(reallabels)
        realTimePeriodAbnormalPds = getDataFramesFromPos(l2l3predetectresultpd, beginpos_endpos_list)

    # 预测标签列表值
    prelabels = list(l2l3predetectresultpd[preflagName])
    beginpos_endpos_list = findAbnormalPos(prelabels)
    preTimePeriodAbnormalPds = getDataFramesFromPos(l2l3predetectresultpd, beginpos_endpos_list)
    # 用于json格式输出的Dict，包含时间段格式
    outputTimePeriodList = []
    for iprepd in preTimePeriodAbnormalPds:
        oneTimePeriodDict = {}
        oneTimePeriodDict["begin_time"] = iprepd[TIME_COLUMN_NAME].iloc[0]  # 预测开始时间
        oneTimePeriodDict["end_time"] = iprepd[TIME_COLUMN_NAME].iloc[-1]  # 预测结束时间
        oneTimePeriodDict["error_info"] = getErrorInfo(iprepd, preflagName, probabilityName)
        oneTimePeriodDict["kpi"] = getKPIInfo(iprepd, preflagName)
        oneTimePeriodDict["performance_var"] = getperformance_var(iprepd, preflagName, probabilityName)
        # 将这个时间段中的数据添加
        outputTimePeriodList.append(oneTimePeriodDict)
    return outputTimePeriodList


# ============================================================================== 对准确率进行统计

"""
函数功能：得到这个一个预测数据的准确率

所有的预测中，我只要预测到就算准确。
isFuzzy=True会将 cpu预测为cpu的内存的预测为内存的都算正确
"""


def getDetectionAccuract(realflags: List[int], preflags: List[List[int]], excludeflags=None,
                         isFuzzy: bool = False) -> float:
    # 判断预判和实际是否相同
    FuzzyFlagDict = {
        "cpu": {10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 34, 35, 80, 81, 82, 83, 84, 85},
        "memory": {50, 51, 52, 53, 54, 55, 60, 61, 62, 63, 64, 65},
    }

    def isright(onerealflag, onepreflagList):
        if isFuzzy:
            # cpu异常被预测为cpu异常
            if onerealflag in FuzzyFlagDict["cpu"] and len(set(onepreflagList) & FuzzyFlagDict["cpu"]) != 0:
                return True
            if onerealflag in FuzzyFlagDict["memory"] and len(set(onepreflagList) & FuzzyFlagDict["memory"]) != 0:
                return True
        return (onerealflag // 10) * 10 in onepreflagList or onerealflag in onepreflagList

    if excludeflags is None:
        excludeflags = []
    assert len(realflags) == len(preflags)
    # 得到预测对的数量 # 数值相等或者数值的百分比
    allnumber = 0
    rightnumber = 0
    for i in range(0, len(realflags)):
        if realflags[i] in excludeflags:
            continue
        allnumber += 1
        if isright(realflags[i], preflags[i]):
            rightnumber += 1
    if allnumber == 0:
        return -1
    return rightnumber / allnumber


"""
得到每个异常的召回率、精确率等
"""


def getDetectionRecallPrecision(realflags: List[int], preflags: List[List[int]], abnormalsSet: Set, spath=None) -> Dict:
    # 计算调和平均数
    def harmonic_mean(data):  # 计算调和平均数
        total = 0
        for i in data:
            if i == 0:  # 处理包含0的情况
                return 0
            total += 1 / i
        return len(data) / total

    assert len(realflags) == len(preflags)
    rightflagSet = set([(i // 10) * 10 for i in abnormalsSet]) | abnormalsSet

    real_abnormalnums = 0  # 异常的总数量
    pre_allabnormalnums = 0  # 所有预测数据中，被预测为异常的数量
    abnormal_rightabnormal_nums = 0  # 异常被预测为正确的个数
    abnormal_abnormal_nums = 0  # 异常被预测为!=0的数量
    abnormal_normal_nums = 0  # 异常被预测为正常的数量
    abnormal_memory_nums = 0  # 异常被预测为内存异常的数量
    abnormal_cpu_nums = 0  # 异常被预测为cpu异常的数

    for i in range(len(realflags)):
        if realflags[i] in abnormalsSet:
            real_abnormalnums += 1  # 表示异常的真实数量+1
        # 如果这个时间点预测的异常和我们需要的异常
        if len(set(preflags[i]) & rightflagSet) != 0:
            pre_allabnormalnums += 1  # 被预测为异常的真实数量+1

        if realflags[i] in abnormalsSet:
            #  现在实际预测值是异常
            if preflags[i] == [0]:
                abnormal_normal_nums += 1
            if preflags[i] != [0]:
                abnormal_abnormal_nums += 1
            # 判断预测的点是否包含了异常
            if len(set(preflags[i]) & rightflagSet) != 0:
                abnormal_rightabnormal_nums += 1  # 异常预测正确
            if len(set(preflags[i]) & CPU_ABNORMAL_TYPE) != 0:
                abnormal_cpu_nums += 1
            if len(set(preflags[i]) & MEMORY_ABNORMAL_TYPE) != 0:
                abnormal_memory_nums += 1
    infoDict = {"num": real_abnormalnums,
                "recall": -1 if real_abnormalnums == 0 else abnormal_rightabnormal_nums / real_abnormalnums,
                "precison": -1 if pre_allabnormalnums == 0 else abnormal_rightabnormal_nums / pre_allabnormalnums,
                "per_abnormal": -1 if real_abnormalnums == 0 else abnormal_abnormal_nums / real_abnormalnums,
                "per_normal": -1 if real_abnormalnums == 0 else abnormal_normal_nums / real_abnormalnums,
                "cpu_abnormal": -1 if real_abnormalnums == 0 else abnormal_cpu_nums / real_abnormalnums,
                "memory_abnormal": -1 if real_abnormalnums == 0 else abnormal_memory_nums / real_abnormalnums}
    infoDict["f-score"] = harmonic_mean([infoDict["recall"], infoDict["precison"]])
    return infoDict


"""
    对预测结果进行分析得到准确率之类的信息
"""


def analysePredictResult(predictpd: pd.DataFrame, spath: str, windowsize: int = 3):
    predictpd = predictpd.copy()
    # 首先去除某些不需要的
    # predictpd = remove_Abnormal_Head_Tail(predictpd, windowsize=windowsize, abnormals={
    #     41, 42, 43, 44, 45,
    #     71, 72, 73, 74, 75,
    #     91, 92, 93, 94, 95, 99,
    # }
    # 去除每个异常的首尾
    predictpd = removeAllHeadTail(predictPd=predictpd, windowsize=windowsize)
    analyseDict = {}
    analyseDict[0] = getDetectionRecallPrecision(predictpd["faultFlag"].tolist(), predictpd["preFlag"].tolist(), {0})
    analyseDict[10] = getDetectionRecallPrecision(predictpd["faultFlag"].tolist(), predictpd["preFlag"].tolist(),
                                                  {11, 12, 13, 14, 15})
    analyseDict[20] = getDetectionRecallPrecision(predictpd["faultFlag"].tolist(), predictpd["preFlag"].tolist(),
                                                  {21, 22, 23, 24, 25})
    analyseDict[30] = getDetectionRecallPrecision(predictpd["faultFlag"].tolist(), predictpd["preFlag"].tolist(),
                                                  {31, 32, 33, 34, 35})
    analyseDict[50] = getDetectionRecallPrecision(predictpd["faultFlag"].tolist(), predictpd["preFlag"].tolist(),
                                                  {51, 52, 53, 54, 55})
    analyseDict[60] = getDetectionRecallPrecision(predictpd["faultFlag"].tolist(), predictpd["preFlag"].tolist(),
                                                  {61, 62, 63, 64, 65})
    analyseDict[80] = getDetectionRecallPrecision(predictpd["faultFlag"].tolist(), predictpd["preFlag"].tolist(),
                                                  {81, 82, 83, 84, 85})
    analyseDict[90] = getDetectionRecallPrecision(predictpd["faultFlag"].tolist(), predictpd["preFlag"].tolist(),
                                                  {91, 92, 93, 94, 95})
    analyseDict["cpu"] = getDetectionRecallPrecision(predictpd["faultFlag"].tolist(), predictpd["preFlag"].tolist(), {
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35,
        81, 82, 83, 84, 85,
    })
    analyseDict["memory"] = getDetectionRecallPrecision(predictpd["faultFlag"].tolist(), predictpd["preFlag"].tolist(),
                                                        {
                                                            51, 52, 53, 54, 55,
                                                            61, 62, 63, 64, 65,
                                                        })
    analyseDict[111] = getDetectionRecallPrecision(predictpd["faultFlag"].tolist(), predictpd["preFlag"].tolist(),
                                                   {111})
    analyseDict[121] = getDetectionRecallPrecision(predictpd["faultFlag"].tolist(), predictpd["preFlag"].tolist(),
                                                   {121})
    analyseDict[131] = getDetectionRecallPrecision(predictpd["faultFlag"].tolist(), predictpd["preFlag"].tolist(),
                                                   {131})
    analyseDict[132] = getDetectionRecallPrecision(predictpd["faultFlag"].tolist(), predictpd["preFlag"].tolist(),
                                                   {132})
    analyseDict[141] = getDetectionRecallPrecision(predictpd["faultFlag"].tolist(), predictpd["preFlag"].tolist(),
                                                   {141})
    analyseDict[161] = getDetectionRecallPrecision(predictpd["faultFlag"].tolist(), predictpd["preFlag"].tolist(),
                                                   {161})

    accuracy_normal = getDetectionAccuract(realflags=predictpd["faultFlag"].tolist(),
                                           preflags=predictpd["preFlag"].tolist())
    accuracy_nonormal = getDetectionAccuract(realflags=predictpd["faultFlag"].tolist(),
                                             preflags=predictpd["preFlag"].tolist(), excludeflags={0})
    accuracy_nonormal_fuzzy = getDetectionAccuract(realflags=predictpd["faultFlag"].tolist(),
                                                   preflags=predictpd["preFlag"].tolist(), excludeflags={0},
                                                   isFuzzy=True)

    # ===================================== 将信息进行保存
    if spath is not None:
        tpd = pd.DataFrame(data=analyseDict).T
        savepdfile(tpd, spath, "统计数据.csv", index=True)
        # 写入准确率信息
        wrfteinfo = [
            "1. 包含正常准确率: {:.2%}\n".format(accuracy_normal),
            "2. 去除正常准确率: {:.2%}\n".format(accuracy_nonormal),
            "3. 去除正常模糊准确率: {:.2%}\n".format(accuracy_nonormal_fuzzy),
        ]
        with open(os.path.join(spath, "4.准确率.txt"), "w", encoding="utf-8") as f:
            f.writelines(wrfteinfo)
