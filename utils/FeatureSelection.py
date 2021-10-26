"""
这个文件的本意是包含特征提取的相关函数
1. Kstest相比较的函数以及Benjamini-Yakutieli Procedure的函数
2. 比较两个相同类型的dataFrame，然后得到pvalue
"""
from typing import Tuple, List, Dict, Union, Any
import utils.DataFrameOperation
import pandas as pd
import numpy as np
from utils.DefineData import *
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

"""
函数功能：传进来两个DataFrame 得到对应特征的pvalue
返回值：是对应特征名字 : pvalude值
"""


def getPvalueFromTwoDataFrame(dataFrame1: pd.DataFrame, dataFrame2: pd.DataFrame) -> Union[
    Tuple[None, bool],Tuple[Dict[Any, float], bool]]:
    # 首先保证两个数据
    if not utils.DataFrameOperation.judgeSameFrames([dataFrame1, dataFrame2]):
        return None, True

    resDict = {}
    for featurename in dataFrame1.columns.array:
        if featurename in EXCLUDE_FEATURE_NAME:
            continue
        tPvalue = getCDFPvalueFromTwoColumns(list(dataFrame1[featurename]), list(dataFrame2[featurename]))
        resDict[featurename] = tPvalue
    return resDict, False


"""
函数功能：获得两个列表的Pvalue
"""


def getCDFPvalueFromTwoColumns(l1: List[float], l2: List[float]) -> float:
    list1 = l1
    list2 = l2
    cdf1 = sm.distributions.ECDF(list1)
    cdf2 = sm.distributions.ECDF(list2)
    minValueX = min(min(list1), min(list2))
    maxValueX = max(max(list1), max(list2))
    xline = np.linspace(minValueX, maxValueX + 1)
    y1 = cdf1(xline)
    y2 = cdf2(xline)
    return stats.kstest(y1, y2)[1]


"""
函数功能：通过Benjamini_Yekutieli 对PValue 进行选择
传入进来的是一个字典，特征名：pvalue
返回值是两个字典，第一个是选中的特征名字:pvalue  第二个是未选中的特征名字:pvalue
"""


def getFeatureNameByBenjamini_Yekutiel(dictPvalues: Dict[str, float]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    lfeaturenames = list(dictPvalues.keys())
    lpvalues = list(dictPvalues.values())
    isReject, pvals_corrected, _, _ = multipletests(lpvalues, alpha=FDR, method='fdr_by', is_sorted=False,
                                                    returnsorted=False)

    selectFeature = {}  # 认为特征不同，所以选中作为主要的特征来进行训练模型
    notselectFeature = {}  # 认为两者的特征相同，所以排除这些特征
    for i, flag in enumerate(isReject):
        nowfeaturename = lfeaturenames[i]
        if flag:
            # 结果为True， 表示拒绝接受两者是相同的
            selectFeature[nowfeaturename] = pvals_corrected[i]
        else:
            notselectFeature[nowfeaturename] = pvals_corrected[i]
    return selectFeature, notselectFeature


"""
函数功能：将一个标准DataFrame传入进去，其他的都是比较DataFrame，得到所有包含所有有效特征值的列表
函数返回值：包含所有选中特征的一个大的DataFrame以及所有正常数据和异常数据
"""


def getUsefulFeatureFromAllDataFrames(normalpd: pd.DataFrame, abnormalpd: List[pd.DataFrame]) -> Union[
    Tuple[None, bool], Tuple[pd.DataFrame, bool]]:
    allPdList = [normalpd]
    allPdList.extend(abnormalpd)

    # 判断列表中是否都有相同的数据结构
    if not utils.DataFrameOperation.judgeSameFrames(allPdList):
        return None, True

    diffset = set()
    for ipd in abnormalpd:
        #  使用KsTest 求出pvalue
        dicFeaturePvalue, err = getPvalueFromTwoDataFrame(normalpd, ipd)
        if err:
            return None, True
        # Benjamini_Yekutieli 选择需要的特征
        selectFeatureDict, _ = getFeatureNameByBenjamini_Yekutiel(dicFeaturePvalue)

        selectFeatureSet = set(selectFeatureDict.keys())

        diffset = set.union(diffset, selectFeatureSet)

    userfulFeatureList = list(diffset)
    if FAULT_FLAG in normalpd.columns.array:
        userfulFeatureList.append(FAULT_FLAG)
    if TIME_COLUMN_NAME in normalpd.columns.array:
        userfulFeatureList.append(TIME_COLUMN_NAME)
    # 得到包含所有有用特征的列表
    userfulFeatureList.sort()
    # 存放使用有用特征列表的DataFrame
    allPdList = [ipd[userfulFeatureList] for ipd in allPdList]

    # 合并所有的列表得到一个训练的DataFrame
    mergedPD, err = utils.DataFrameOperation.mergeDataFrames(allPdList)
    if err:
        return None, True

    #  将DataFrame中的某些标签移动
    mergedPD = utils.DataFrameOperation.SortLabels(mergedPD)
    mergedPD = utils.DataFrameOperation.PushLabelToFirst(mergedPD, label=TIME_COLUMN_NAME)
    mergedPD = utils.DataFrameOperation.PushLabelToEnd(mergedPD, label=FAULT_FLAG)

    return mergedPD, False


"""
选择有用的标签
"""

def getUsefulFeatureFromNormalAndAbnormal(normalpd: pd.DataFrame, abnormalpd: List[pd.DataFrame]) -> Union[
    tuple[None, bool], tuple[list, bool]]:
    allPdList = [normalpd]
    allPdList.extend(abnormalpd)

    # 判断列表中是否都有相同的数据结构
    if not utils.DataFrameOperation.judgeSameFrames(allPdList):
        return None, True

    diffset = set()
    for ipd in abnormalpd:
        #  使用KsTest 求出pvalue
        dicFeaturePvalue, err = getPvalueFromTwoDataFrame(normalpd, ipd)
        if err:
            return None, True
        # Benjamini_Yekutieli 选择需要的特征
        selectFeatureDict, _ = getFeatureNameByBenjamini_Yekutiel(dicFeaturePvalue)

        selectFeatureSet = set(selectFeatureDict.keys())

        diffset = set.union(diffset, selectFeatureSet)

    userfulFeatureList = list(diffset)
    userfulFeatureList.sort()
    return userfulFeatureList, False

