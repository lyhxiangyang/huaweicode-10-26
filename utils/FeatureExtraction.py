"""
这个文件的本意是包含特征提取的函数
比如：
1. 将滑动窗口设置，然后提取最小值、最大值等数值
"""
from numpy import nan
from typing import Tuple, Union, List, Any

import pandas as pd

from utils.DataFrameOperation import SortLabels, PushLabelToFirst, PushLabelToEnd
from utils.DefineData import *

"""
函数功能：将一个DataFrame的结构中的数据按照滑动窗口大小提取如下特征值
# - 最小值
# - 最大值
# - 5%大的值
# - 25% 大的值
# - 50%大的值
# - 75%大的值
# - 95%大的值
# - 平均数
# - 方差
# - 倾斜度
# - 峰度

备注：保证传入进来的DataFrame的FaultFlag只有一类
"""


def featureExtraction(featurePD: pd.DataFrame, windowSize: int = 5) -> Union[
    Tuple[None, bool], Tuple[Union[pd.DataFrame, pd.Series], bool]]:
    # 1个特征会生成很多新的特征, 下面是这个特征需要添加的后缀名
    suffix_name = ["_min", "_max", "_percentage_5", "_percentage_25", "_percentage_50", "_percentage_75",
                   "_percentage_95", "_mean", "_var", "_std", "_skewness", "_kurtosis"]
    # 查分的后缀名 上面suffix_name中的_diff是需要的，用来在字典中生成对应的keys
    Diff_suffix = "_diff"

    # 一个内部函数用来获得列表最后一位
    def getListEnd(list1: List):
        if len(list1) == 0:
            return 0
        return list1[-1]

    # 长度为0 不进行处理
    if len(featurePD) == 0:
        return None, True

    nowFaultFlag = featurePD[FAULT_FLAG][0]

    # 获得下一个窗口大小得函数
    def getnext(beginpos: int) -> Tuple[int, int]:
        endpos = beginpos + windowSize
        if endpos > len(featurePD):
            endpos = len(featurePD)
        return beginpos, endpos

    # 保存结果的返回值
    resDataFrame = pd.DataFrame()

    for featurename in featurePD.columns.array:
        # 先去除掉要排除的特征值
        if featurename in EXCLUDE_FEATURE_NAME:
            continue

        # 特征名字featurename
        # 接下来 创建一个字典，对应每个特征名字，value是一个数组
        # 为saveTable添加列项
        myColumeNamesList = [featurename + suistr for suistr in suffix_name]
        # myColumeNamesValues = [[]] * len(myColumeNames) #这里面存储的都是链表, 比如上面长度为2，那么这个值为[[], []]
        myColumeNamesDict = dict(zip(myColumeNamesList, [[] for _ in range(len(myColumeNamesList))]))

        beginLine = 0
        # 接下来按照滑动窗口大小，将对应的特征值计算一遍
        while beginLine + windowSize <= len(featurePD):
            # 获得特征值中对应滑动窗口大小的数值。

            beginLine, endLine = getnext(beginLine)
            # print(beginLine, endLine)
            # 获得对应一列的数据
            calSerials = featurePD.iloc[beginLine:endLine][featurename]
            # print(list(calSerials))

            newfeatureName = featurename + "_min"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.min()
            # 判断是否有key的存在，不存在就新建
            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            # print(newfeatureName, calSerials.min())
            if newfeatureName is None:
                return None, True

            newfeatureName = featurename + "_max"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.max()

            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            if newfeatureName is None:
                return None, True

            newfeatureName = featurename + "_percentage_5"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.quantile(0.05)
            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            if newfeatureName is None:
                return None, True

            newfeatureName = featurename + "_percentage_25"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.quantile(0.25)
            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            if newfeatureName is None:
                return None, True

            newfeatureName = featurename + "_percentage_50"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.quantile(0.5)
            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            if newfeatureName is None:
                return None, True

            newfeatureName = featurename + "_percentage_75"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.quantile(0.75)
            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            if newfeatureName is None:
                return None, True

            newfeatureName = featurename + "_percentage_95"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.quantile(0.95)
            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            if newfeatureName is None:
                return None, True

            newfeatureName = featurename + "_var"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.var()
            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            if newfeatureName is None:
                return None, True

            newfeatureName = featurename + "_std"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.std()
            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            if newfeatureName is None:
                return None, True

            newfeatureName = featurename + "_mean"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.mean()
            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            if newfeatureName is None:
                return None, True

            newfeatureName = featurename + "_skewness"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.skew()
            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            if newfeatureName is None:
                return None, True

            newfeatureName = featurename + "_kurtosis"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.kurtosis()
            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            if newfeatureName is None:
                return None, True

            # 修改起始行号
            beginLine = endLine

        # 我们差分特征的第一个选项是有问题的默认和第一个值一样，我们将其调整成为和第二一样
        for ikey, ilist in myColumeNamesDict.items():
            if ikey.endswith(Diff_suffix) and len(ilist) > 2:
                ilist[0] = ilist[1]

        # 将搜集到的这个特征的信息保存到新的DataFrame中
        # for newfeatureName in myColumeNamesList:
        #     resDataFrame[newfeatureName] = myColumeNamesDict[newfeatureName]
        tDF = pd.DataFrame(myColumeNamesDict)
        # if isEmptyInDataFrame(tDF):
        #     print("2. DataFrame is None")
        #     print("特征名字：", featurename)
        #     tDF.to_csv("tmp/1.error.csv")
        #     exit(1)

        resDataFrame = pd.concat([resDataFrame, tDF], axis=1)
        # if isEmptyInDataFrame(resDataFrame):
        #     print("3. DataFram is None")
    # 为新的DataFrame添加标签
    td = {FAULT_FLAG: [nowFaultFlag for i in range(0, len(resDataFrame))]}
    tpd = pd.DataFrame(td)
    resDataFrame = pd.concat([resDataFrame, tpd], axis=1)

    # 将结果排一下顺序
    resDataFrame = SortLabels(resDataFrame)
    resDataFrame = PushLabelToFirst(resDataFrame, label=TIME_COLUMN_NAME)
    resDataFrame = PushLabelToEnd(resDataFrame, label=FAULT_FLAG)

    if DEBUG:
        print("featureExtraction".center(40, "*"))
        print(resDataFrame.iloc[:, 0:2])
        print("end".center(40, "*"))

    resDataFrame.fillna(0, inplace=True)

    return resDataFrame, False


"""
函数功能：将一个DataFrame的结构中的数据按照滑动窗口大小提取如下特征值
# - 最小值
# - 最大值
# - 5%大的值
# - 25% 大的值
# - 50%大的值
# - 75%大的值
# - 95%大的值
# - 平均数
# - 方差
# - 倾斜度
# - 峰度

备注：保证传入进来的DataFrame的FaultFlag只有一类
"""


def featureExtraction_excludeAccumulation(featurePD: pd.DataFrame, windowSize: int = 5,
                                          accumulateFeature: List[str] = []) -> Union[
    Tuple[None, bool], Tuple[Union[pd.DataFrame, pd.Series], bool]]:
    # 1个特征会生成很多新的特征, 下面是这个特征需要添加的后缀名
    suffix_name = ["_min", "_max", "_percentage_5", "_percentage_25", "_percentage_50", "_percentage_75",
                   "_percentage_95", "_mean", "_var", "_std", "_skewness", "_kurtosis"]
    # 查分的后缀名 上面suffix_name中的_diff是需要的，用来在字典中生成对应的keys
    Diff_suffix = "_diff"

    # 一个内部函数用来获得列表最后一位
    def getListEnd(list1: List):
        if len(list1) == 0:
            return 0
        return list1[-1]

    # 长度为0 不进行处理
    if len(featurePD) == 0:
        return None, True

    nowFaultFlag = featurePD[FAULT_FLAG][0]

    # 获得下一个窗口大小得函数
    def getnext(beginpos: int) -> Tuple[int, int]:
        endpos = beginpos + windowSize
        if endpos > len(featurePD):
            endpos = len(featurePD)
        return beginpos, endpos

    # 保存结果的返回值
    resDataFrame = pd.DataFrame()

    for featurename in featurePD.columns.array:
        # 先去除掉要排除的特征值
        if featurename in EXCLUDE_FEATURE_NAME:
            continue

        # 特征名字featurename
        # 接下来 创建一个字典，对应每个特征名字，value是一个数组
        # 为saveTable添加列项
        myColumeNamesList = [featurename + suistr for suistr in suffix_name]
        # myColumeNamesValues = [[]] * len(myColumeNames) #这里面存储的都是链表, 比如上面长度为2，那么这个值为[[], []]
        myColumeNamesDict = dict(zip(myColumeNamesList, [[] for _ in range(len(myColumeNamesList))]))

        beginLine = 0
        # 接下来按照滑动窗口大小，将对应的特征值计算一遍
        while beginLine + windowSize <= len(featurePD):
            # 获得特征值中对应滑动窗口大小的数值。

            beginLine, endLine = getnext(beginLine)
            # print(beginLine, endLine)
            # 获得对应一列的数据
            calSerials = featurePD.iloc[beginLine:endLine][featurename]
            # print(list(calSerials))

            newfeatureName = featurename + "_min"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.min()
            # 判断是否有key的存在，不存在就新建
            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            # print(newfeatureName, calSerials.min())
            if newfeatureName is None:
                return None, True

            newfeatureName = featurename + "_max"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.max()

            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            if newfeatureName is None:
                return None, True

            newfeatureName = featurename + "_percentage_5"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.quantile(0.05)
            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            if newfeatureName is None:
                return None, True

            newfeatureName = featurename + "_percentage_25"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.quantile(0.25)
            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            if newfeatureName is None:
                return None, True

            newfeatureName = featurename + "_percentage_50"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.quantile(0.5)
            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            if newfeatureName is None:
                return None, True

            newfeatureName = featurename + "_percentage_75"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.quantile(0.75)
            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            if newfeatureName is None:
                return None, True

            newfeatureName = featurename + "_percentage_95"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.quantile(0.95)
            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            if newfeatureName is None:
                return None, True

            newfeatureName = featurename + "_var"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.var()
            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            if newfeatureName is None:
                return None, True

            newfeatureName = featurename + "_std"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.std()
            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            if newfeatureName is None:
                return None, True

            newfeatureName = featurename + "_mean"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.mean()
            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            if newfeatureName is None:
                return None, True

            newfeatureName = featurename + "_skewness"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.skew()
            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            if newfeatureName is None:
                return None, True

            newfeatureName = featurename + "_kurtosis"
            newfeatureNameDiff = newfeatureName + Diff_suffix
            featurevalue = calSerials.kurtosis()
            if newfeatureNameDiff not in myColumeNamesDict.keys():
                myColumeNamesDict[newfeatureNameDiff] = []
            myColumeNamesDict[newfeatureNameDiff].append(
                featurevalue - getListEnd(myColumeNamesDict[newfeatureName]))
            myColumeNamesDict[newfeatureName].append(featurevalue)
            if newfeatureName is None:
                return None, True

            # 修改起始行号
            beginLine = endLine

        # 我们差分特征的第一个选项是有问题的默认和第一个值一样，我们将其调整成为和第二一样
        for ikey, ilist in myColumeNamesDict.items():
            if ikey.endswith(Diff_suffix) and len(ilist) > 2:
                ilist[0] = ilist[1]

        # 将搜集到的这个特征的信息保存到新的DataFrame中
        # for newfeatureName in myColumeNamesList:
        #     resDataFrame[newfeatureName] = myColumeNamesDict[newfeatureName]
        tDF = pd.DataFrame(myColumeNamesDict)
        # if isEmptyInDataFrame(tDF):
        #     print("2. DataFrame is None")
        #     print("特征名字：", featurename)
        #     tDF.to_csv("tmp/1.error.csv")
        #     exit(1)

        resDataFrame = pd.concat([resDataFrame, tDF], axis=1)
        # if isEmptyInDataFrame(resDataFrame):
        #     print("3. DataFram is None")
    # 为新的DataFrame添加标签
    td = {FAULT_FLAG: [nowFaultFlag for i in range(0, len(resDataFrame))]}
    tpd = pd.DataFrame(td)
    resDataFrame = pd.concat([resDataFrame, tpd], axis=1)

    # 将结果排一下顺序
    resDataFrame = SortLabels(resDataFrame)
    resDataFrame = PushLabelToFirst(resDataFrame, label=TIME_COLUMN_NAME)
    resDataFrame = PushLabelToEnd(resDataFrame, label=FAULT_FLAG)

    if DEBUG:
        print("featureExtraction".center(40, "*"))
        print(resDataFrame.iloc[:, 0:2])
        print("end".center(40, "*"))

    return resDataFrame, False

"""
保证这个df的时间序列是连续的，并且可能包含多个错误类型
保证带有time 和标签特征
"""


def featureExtractionUsingFeatures(df: pd.DataFrame, windowSize: int = 5, windowRealSize: int = 1, silidWindows: bool = True,
                      extraFeature=None) -> \
        Union[dict[int, dict], Any]:
    if extraFeature is None:
        extraFeature = []
    lendf = len(df)
    resFaulty_PDDict = {}
    resPD = {}
    if windowSize > lendf:
        return resFaulty_PDDict

    # suffix_name = ["_min", "_max", "_percentage_5", "_percentage_25", "_percentage_50", "_percentage_75",
    #                "_percentage_95", "_mean", "_var", "_std", "_skewness", "_kurtosis"]
    # # 查分的后缀名 上面suffix_name中的_diff是需要的，用来在字典中生成对应的keys
    # Diff_suffix = "_diff"
    # # 得到所有的特征值
    mycolumnslist = list(df.columns.array)

    # mycolumns = [ic + isuffix for ic in mycolumns for isuffix in suffix_name]
    # mycolumns.extend([i + Diff_suffix for i in mycolumns])

    # def getRealLabel(labels: pd.Series) -> int:
    #     for i in labels:
    #         if i != 0:
    #             return i
    #     return 0
    def getRealLabel(labels: pd.Series) -> Tuple[int, int]:
        inum = 0
        flag = 0
        for i in labels:
            if i != 0:
                flag = i
                inum += 1
        # 先返回异常的数量，再返回异常的种类
        return inum, flag

    def getListEnd(list1: List):
        if len(list1) == 0:
            return 0
        return list1[-1]

    beginLineNumber = 0
    endLineNumber = windowSize

    while endLineNumber <= lendf:

        tpd = df.iloc[beginLineNumber:endLineNumber, :]
        nowtime = tpd.loc[beginLineNumber, TIME_COLUMN_NAME]
        abnormalNum, realLabel = getRealLabel(tpd.loc[:, FAULT_FLAG])
        # 用来跳过中间阶段
        if realLabel != 0 and abnormalNum < windowRealSize:
            if not silidWindows:
                beginLineNumber += windowSize
                endLineNumber += windowSize
            else:
                beginLineNumber += 1
                endLineNumber += 1
            continue

        if realLabel not in resFaulty_PDDict:
            resFaulty_PDDict[realLabel] = {}
        # 添加时间
        if TIME_COLUMN_NAME not in resFaulty_PDDict[realLabel]:
            resFaulty_PDDict[realLabel][TIME_COLUMN_NAME] = []
        if FAULT_FLAG not in resFaulty_PDDict[realLabel]:
            resFaulty_PDDict[realLabel][FAULT_FLAG] = []
        resFaulty_PDDict[realLabel][TIME_COLUMN_NAME].append(nowtime)
        resFaulty_PDDict[realLabel][FAULT_FLAG].append(realLabel)
        if TIME_COLUMN_NAME not in resPD:
            resPD[TIME_COLUMN_NAME] = []
        if FAULT_FLAG not in resPD:
            resPD[FAULT_FLAG] = []
        resPD[TIME_COLUMN_NAME].append(nowtime)
        resPD[FAULT_FLAG].append(realLabel)

        # 对每个特征进行选择
        for featurename in mycolumnslist:
            if featurename not in extraFeature:
                continue
            if featurename == TIME_COLUMN_NAME or featurename == FAULT_FLAG:
                continue

            calSerials = tpd.loc[:, featurename]

            # min min_diff
            newfeatureName = featurename + "_min"
            newfeatureNameDiff = newfeatureName + "_diff"
            if newfeatureName not in resFaulty_PDDict[realLabel]:
                resFaulty_PDDict[realLabel][newfeatureName] = []
            if newfeatureNameDiff not in resFaulty_PDDict[realLabel]:
                resFaulty_PDDict[realLabel][newfeatureNameDiff] = []
            if newfeatureName not in resPD:
                resPD[newfeatureName] = []
            if newfeatureNameDiff not in resPD:
                resPD[newfeatureNameDiff] = []
            featurevalue = int(calSerials.min())
            featurevaluediff = featurevalue - getListEnd(resFaulty_PDDict[realLabel][newfeatureName])
            resFaulty_PDDict[realLabel][newfeatureName].append(featurevalue)
            resFaulty_PDDict[realLabel][newfeatureNameDiff].append(featurevaluediff)
            resPD[newfeatureNameDiff].append(featurevalue - getListEnd(resPD[newfeatureName]))
            resPD[newfeatureName].append(featurevalue)
            # max max_diff
            newfeatureName = featurename + "_max"
            newfeatureNameDiff = newfeatureName + "_diff"
            if newfeatureName not in resFaulty_PDDict[realLabel]:
                resFaulty_PDDict[realLabel][newfeatureName] = []
            if newfeatureNameDiff not in resFaulty_PDDict[realLabel]:
                resFaulty_PDDict[realLabel][newfeatureNameDiff] = []
            if newfeatureName not in resPD:
                resPD[newfeatureName] = []
            if newfeatureNameDiff not in resPD:
                resPD[newfeatureNameDiff] = []
            featurevalue = int(calSerials.max())
            featurevaluediff = featurevalue - getListEnd(resFaulty_PDDict[realLabel][newfeatureName])
            resFaulty_PDDict[realLabel][newfeatureName].append(featurevalue)
            resFaulty_PDDict[realLabel][newfeatureNameDiff].append(featurevaluediff)
            resPD[newfeatureNameDiff].append(featurevalue - getListEnd(resPD[newfeatureName]))
            resPD[newfeatureName].append(featurevalue)

            # percentage_50
            newfeatureName = featurename + "_percentage_50"
            newfeatureNameDiff = newfeatureName + "_diff"
            if newfeatureName not in resFaulty_PDDict[realLabel]:
                resFaulty_PDDict[realLabel][newfeatureName] = []
            if newfeatureNameDiff not in resFaulty_PDDict[realLabel]:
                resFaulty_PDDict[realLabel][newfeatureNameDiff] = []
            if newfeatureName not in resPD:
                resPD[newfeatureName] = []
            if newfeatureNameDiff not in resPD:
                resPD[newfeatureNameDiff] = []
            featurevalue = int(calSerials.quantile(0.5))
            featurevaluediff = featurevalue - getListEnd(resFaulty_PDDict[realLabel][newfeatureName])
            resFaulty_PDDict[realLabel][newfeatureName].append(featurevalue)
            resFaulty_PDDict[realLabel][newfeatureNameDiff].append(featurevaluediff)
            resPD[newfeatureNameDiff].append(featurevalue - getListEnd(resPD[newfeatureName]))
            resPD[newfeatureName].append(featurevalue)

            # var
            newfeatureName = featurename + "_var"
            newfeatureNameDiff = newfeatureName + "_diff"
            if newfeatureName not in resFaulty_PDDict[realLabel]:
                resFaulty_PDDict[realLabel][newfeatureName] = []
            if newfeatureNameDiff not in resFaulty_PDDict[realLabel]:
                resFaulty_PDDict[realLabel][newfeatureNameDiff] = []
            if newfeatureName not in resPD:
                resPD[newfeatureName] = []
            if newfeatureNameDiff not in resPD:
                resPD[newfeatureNameDiff] = []
            featurevalue = int(calSerials.var())
            featurevaluediff = featurevalue - getListEnd(resFaulty_PDDict[realLabel][newfeatureName])
            resFaulty_PDDict[realLabel][newfeatureName].append(featurevalue)
            resFaulty_PDDict[realLabel][newfeatureNameDiff].append(featurevaluediff)
            resPD[newfeatureNameDiff].append(featurevalue - getListEnd(resPD[newfeatureName]))
            resPD[newfeatureName].append(featurevalue)
            # std
            newfeatureName = featurename + "_std"
            newfeatureNameDiff = newfeatureName + "_diff"
            if newfeatureName not in resFaulty_PDDict[realLabel]:
                resFaulty_PDDict[realLabel][newfeatureName] = []
            if newfeatureNameDiff not in resFaulty_PDDict[realLabel]:
                resFaulty_PDDict[realLabel][newfeatureNameDiff] = []
            if newfeatureName not in resPD:
                resPD[newfeatureName] = []
            if newfeatureNameDiff not in resPD:
                resPD[newfeatureNameDiff] = []
            featurevalue = int(calSerials.std())
            featurevaluediff = featurevalue - getListEnd(resFaulty_PDDict[realLabel][newfeatureName])
            resFaulty_PDDict[realLabel][newfeatureName].append(featurevalue)
            resFaulty_PDDict[realLabel][newfeatureNameDiff].append(featurevaluediff)
            resPD[newfeatureNameDiff].append(featurevalue - getListEnd(resPD[newfeatureName]))
            resPD[newfeatureName].append(featurevalue)

            # mean
            newfeatureName = featurename + "_mean"
            newfeatureNameDiff = newfeatureName + "_diff"
            if newfeatureName not in resFaulty_PDDict[realLabel]:
                resFaulty_PDDict[realLabel][newfeatureName] = []
            if newfeatureNameDiff not in resFaulty_PDDict[realLabel]:
                resFaulty_PDDict[realLabel][newfeatureNameDiff] = []
            if newfeatureName not in resPD:
                resPD[newfeatureName] = []
            if newfeatureNameDiff not in resPD:
                resPD[newfeatureNameDiff] = []
            featurevalue = int(calSerials.mean())
            featurevaluediff = featurevalue - getListEnd(resFaulty_PDDict[realLabel][newfeatureName])
            resFaulty_PDDict[realLabel][newfeatureName].append(featurevalue)
            resFaulty_PDDict[realLabel][newfeatureNameDiff].append(featurevaluediff)
            resPD[newfeatureNameDiff].append(featurevalue - getListEnd(resPD[newfeatureName]))
            resPD[newfeatureName].append(featurevalue)

            # skewness
            newfeatureName = featurename + "_skewness"
            newfeatureNameDiff = newfeatureName + "_diff"
            if newfeatureName not in resFaulty_PDDict[realLabel]:
                resFaulty_PDDict[realLabel][newfeatureName] = []
            if newfeatureNameDiff not in resFaulty_PDDict[realLabel]:
                resFaulty_PDDict[realLabel][newfeatureNameDiff] = []
            if newfeatureName not in resPD:
                resPD[newfeatureName] = []
            if newfeatureNameDiff not in resPD:
                resPD[newfeatureNameDiff] = []
            featurevalue = calSerials.skew()
            if featurevalue is not nan:
                featurevalue = int(featurevalue)
            featurevaluediff = featurevalue - getListEnd(resFaulty_PDDict[realLabel][newfeatureName])
            resFaulty_PDDict[realLabel][newfeatureName].append(featurevalue)
            resFaulty_PDDict[realLabel][newfeatureNameDiff].append(featurevaluediff)
            resPD[newfeatureNameDiff].append(featurevalue - getListEnd(resPD[newfeatureName]))
            resPD[newfeatureName].append(featurevalue)

            # kurtosis
            newfeatureName = featurename + "_kurtosis"
            newfeatureNameDiff = newfeatureName + "_diff"
            if newfeatureName not in resFaulty_PDDict[realLabel]:
                resFaulty_PDDict[realLabel][newfeatureName] = []
            if newfeatureNameDiff not in resFaulty_PDDict[realLabel]:
                resFaulty_PDDict[realLabel][newfeatureNameDiff] = []
            if newfeatureName not in resPD:
                resPD[newfeatureName] = []
            if newfeatureNameDiff not in resPD:
                resPD[newfeatureNameDiff] = []
            featurevalue = calSerials.kurtosis()
            if featurevalue is not nan:
                featurevalue = int(featurevalue)
            featurevaluediff = featurevalue - getListEnd(resFaulty_PDDict[realLabel][newfeatureName])
            resFaulty_PDDict[realLabel][newfeatureName].append(featurevalue)
            resFaulty_PDDict[realLabel][newfeatureNameDiff].append(featurevaluediff)
            resPD[newfeatureNameDiff].append(featurevalue - getListEnd(resPD[newfeatureName]))
            resPD[newfeatureName].append(featurevalue)

        if not silidWindows:
            beginLineNumber += windowSize
            endLineNumber += windowSize
        else:
            beginLineNumber += 1
            endLineNumber += 1

    # # 将所有resDict中的所有数据diff的第一列中的数据替换为第二个
    for ifaulty, featureDict in resFaulty_PDDict.items():
        for ifeaturename, ilist in featureDict.items():
            if not ifeaturename.endswith("_diff"):
                continue
            if len(ilist) >= 2:
                resFaulty_PDDict[ifaulty][ifeaturename][0] = ilist[1]
    for ifeaturename, ilist in resPD.items():
        if not ifeaturename.endswith("_diff"):
            continue
        if len(ilist) >= 2:
            resPD[ifeaturename][0] = ilist[1]

    # 将resDict 转化为 resDFDict
    resDFDict = {}
    for ifaulty, featureDict in resFaulty_PDDict.items():
        resDataFrame = pd.DataFrame(data=featureDict)
        resDataFrame = SortLabels(resDataFrame)
        resDataFrame = PushLabelToFirst(resDataFrame, label=TIME_COLUMN_NAME)
        resDataFrame = PushLabelToEnd(resDataFrame, label=FAULT_FLAG)
        resDataFrame.fillna(0, inplace=True)
        resDFDict[ifaulty] = resDataFrame
    # 原始文件的变化
    originDF = pd.DataFrame(data=resPD)
    originDF = SortLabels(originDF)
    originDF = PushLabelToFirst(originDF, label=TIME_COLUMN_NAME)
    originDF = PushLabelToEnd(originDF, label=FAULT_FLAG)
    originDF.fillna(0, inplace=True)
    # 原始文件的处理， 以及其中的错误码
    return originDF, resDFDict
