"""
这个文件的本意是包含特征提取的相关函数
1. Kstest相比较的函数以及Benjamini-Yakutieli Procedure的函数
2. 比较两个相同类型的dataFrame，然后得到pvalue
"""
from typing import Tuple, List, Dict, Any

import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import utils.DataFrameOperation
from utils.DefineData import *

# ===============================================================================  下面是2021-12-8日写的和KSTest新的有关的函数

"""
从两个列表中相比较，等到PValue
"""


def getPValueFromTwoList(l1: List[float], l2: List[float]) -> float:
    return stats.kstest(l1, l2)[1]


"""
函数功能：比较两个DataFrame之间指定的特征值之间的pvalue 
"""


def getPValueFromTwoDF(dataFrame1: pd.DataFrame, dataFrame2: pd.DataFrame, compareFeatures: List[str] = None) -> Dict:
    if compareFeatures is None:
        # 默认情况下是所有的列
        # 首先保证两个数据特征值是相等的
        assert utils.DataFrameOperation.judgeSameFrames([dataFrame1, dataFrame2])
        compareFeatures = list(dataFrame1.columns)
    # 如果两个DataFrame中有time和faultFlag 那么要去掉
    if TIME_COLUMN_NAME in compareFeatures:
        compareFeatures.remove(TIME_COLUMN_NAME)
    if FAULT_FLAG in compareFeatures:
        compareFeatures.remove(FAULT_FLAG)
    # 判断所选择的特征一定在这两个dataFrame中
    assert set(compareFeatures).issubset(set(dataFrame1.columns))
    assert set(compareFeatures).issubset(set(dataFrame2.columns))

    # 到这一步可以保证所有的特征在两个DataFrame中都存在, 并且不存在time和faultflag
    # ===
    resDict = {}
    for featurename in compareFeatures:
        tPvalue = getPValueFromTwoList(list(dataFrame1[featurename]), list(dataFrame2[featurename]))
        resDict[featurename] = tPvalue
    return resDict


"""
函数功能：通过Benjamini_Yekutieli 对PValue 进行选择
传入进来的是一个字典，特征名：pvalue
返回值是两个字典，第一个是选中的特征名字:pvalue  第二个是未选中的特征名字:pvalue
"""


def getFeatureNameByBenjamini_Yekutiel(dictPvalues: Dict[str, float], fdr: float=0.05) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    lfeaturenames = list(dictPvalues.keys())
    lpvalues = list(dictPvalues.values())
    isReject, pvals_corrected, _, _ = multipletests(lpvalues, alpha=fdr, method='fdr_by', is_sorted=False,
                                                    returnsorted=False)

    selectFeatureDict = {}  # 认为特征不同，所以选中作为主要的特征来进行训练模型
    notselectFeatureDict = {}  # 认为两者的特征相同，所以排除这些特征
    for i, flag in enumerate(isReject):
        nowfeaturename = lfeaturenames[i]
        if flag:
            # 结果为True， 表示拒绝接受两者是相同的
            selectFeatureDict[nowfeaturename] = pvals_corrected[i]
        else:
            notselectFeatureDict[nowfeaturename] = pvals_corrected[i]
    return selectFeatureDict, notselectFeatureDict
