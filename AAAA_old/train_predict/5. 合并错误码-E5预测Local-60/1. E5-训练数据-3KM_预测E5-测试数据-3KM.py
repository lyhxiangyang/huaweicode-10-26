from typing import List

import pandas as pd

from Classifiers.TrainToTest import ModelTrainAndTest
from utils.DataFrameOperation import mergeDataFrames
from utils.DefineData import FAULT_FLAG

trainNormalDataPath = [
    R"D:\HuaweiMachine\huaweicode-10-26\tmp\tData-11-09\E5正常数据\server数据处理\9.特征提取所有错误-处理首尾\0.csv",
]

trainAbnormalDataPath = [
    (60, R"D:\HuaweiMachine\huaweicode-10-26\tmp\tData-11-09\训练数据\多机-E5-server-3KM\9.特征提取所有错误-处理首尾\61.csv"),
    (60, R"D:\HuaweiMachine\huaweicode-10-26\tmp\tData-11-09\训练数据\多机-E5-server-3KM\9.特征提取所有错误-处理首尾\62.csv"),
    (60, R"D:\HuaweiMachine\huaweicode-10-26\tmp\tData-11-09\训练数据\多机-E5-server-3KM\9.特征提取所有错误-处理首尾\63.csv"),
    (60, R"D:\HuaweiMachine\huaweicode-10-26\tmp\tData-11-09\训练数据\多机-E5-server-3KM\9.特征提取所有错误-处理首尾\64.csv"),
    (60, R"D:\HuaweiMachine\huaweicode-10-26\tmp\tData-11-09\训练数据\多机-E5-server-3KM\9.特征提取所有错误-处理首尾\65.csv"),
]

testNormalDataPath = [
    R"D:\HuaweiMachine\huaweicode-10-26\tmp\tData-11-09\测试数据\多机-E5-server-3KM\9.特征提取所有错误-处理首尾\0.csv",
]

testAbnormalDataPath = [
    (60, R"D:\HuaweiMachine\huaweicode-10-26\tmp\tData-11-09\测试数据\多机-E5-server-3KM\9.特征提取所有错误-处理首尾\61.csv"),
    (60, R"D:\HuaweiMachine\huaweicode-10-26\tmp\tData-11-09\测试数据\多机-E5-server-3KM\9.特征提取所有错误-处理首尾\62.csv"),
    (60, R"D:\HuaweiMachine\huaweicode-10-26\tmp\tData-11-09\测试数据\多机-E5-server-3KM\9.特征提取所有错误-处理首尾\63.csv"),
    (60, R"D:\HuaweiMachine\huaweicode-10-26\tmp\tData-11-09\测试数据\多机-E5-server-3KM\9.特征提取所有错误-处理首尾\64.csv"),
    (60, R"D:\HuaweiMachine\huaweicode-10-26\tmp\tData-11-09\测试数据\多机-E5-server-3KM\9.特征提取所有错误-处理首尾\65.csv"),
]

def get_List_pre_suffix(clist: List[str], prefix: str = "", suffix: str = "") -> List[str]:
    return [i for i in clist if i.startswith(prefix) and i.endswith(suffix)]
def get_List_pre_nosuffix(clist: List[str], prefix: str = "", suffix: str = "") -> List[str]:
    return [i for i in clist if i.startswith(prefix) and not i.endswith(suffix)]
def get_List_nosuffix(clist: List[str], suffix: str="") -> List[str]:
    if suffix == "":
        return clist
    return [i for i in clist if not i.endswith(suffix)]

# 将一个DataFrame的FAULT_FLAG重值为ff
def setPDfaultFlag(df: pd.DataFrame, ff: int) -> pd.DataFrame:
    if FAULT_FLAG in df.columns.array:
        df = df.drop(FAULT_FLAG, axis=1)
    lengthpd = len(df)
    ffdict = {FAULT_FLAG: [ff] * lengthpd}
    tpd = pd.DataFrame(data=ffdict)
    tpd = pd.concat([df, tpd], axis=1)
    return tpd

"""
指定标准化之后并且特征提取的数据， 去除后缀为_diff的特征，对强度为15的数据进行验证
"""
# 预测未处理数据
if __name__ == "__main__":
    spath = "tmp/E5多机预测Local-标准化特征提取-合并错误"
    # trainedPDList: list[Union[Union[TextFileReader, Series, DataFrame, None], Any]] = []
    # for i in trainDataPath:
    #     tpd = pd.read_csv(i)
    #     trainedPDList.append(tpd)
    # allTrainedPD, err = mergeDataFrames(trainedPDList)
    # allTrainedPD: pd.DataFrame
    # if err:
    #     print("train合并出错")
    #     exit(1)
    #
    # testPDList = []
    # for i in testDataPath:
    #     tpd = pd.read_csv(i)
    #     testPDList.append(tpd)
    # allTestPD, err = mergeDataFrames(testPDList)
    # if err:
    #     print("test合并出错")
    #     exit(1)
    #==================================================================读取训练的normal数据
    trainNormalList = []
    for i in trainNormalDataPath:
        tpd = pd.read_csv(i)
        trainNormalList.append(tpd)
    #==================================================================读取训练的abnormal数据
    trainAbnormalList = []
    for ilabel, ipath in trainAbnormalDataPath:
        tpd = pd.read_csv(ipath)
        # 修改list的标签
        tpd = setPDfaultFlag(tpd, ilabel)
        trainAbnormalList.append(tpd)
    #==================================================================读取测试的normal数据
    testNormalList = []
    for i in testNormalDataPath:
        tpd = pd.read_csv(i)
        testNormalList.append(tpd)
    #==================================================================读取训练的normal数据
    testAbnormalList = []
    for ilabel, ipath in testAbnormalDataPath:
        tpd = pd.read_csv(ipath)
        tpd = setPDfaultFlag(tpd, ilabel)
        testAbnormalList.append(tpd)

    #==================================================================将所有的训练数据进行合并
    allTrainedPD, err = mergeDataFrames(trainNormalList + trainAbnormalList)
    if err:
        print("训练数据合并失败")
        exit(1)
    #==================================================================将所有的测试数据进行合并
    allTestPD, err = mergeDataFrames(testNormalList + testAbnormalList)
    if err:
        print("测试数据合并失败")
        exit(1)



    # 获得需要训练的特征
    # allfeatureload1_nosuffix = get_List_pre_nosuffix(list(allTrainedPD.columns.array), prefix="used_", suffix="_diff")
    allfeatureload1_nosuffix = get_List_pre_nosuffix(list(allTrainedPD.columns.array), prefix="pgfree_", suffix="_diff")


    print("选择的特征：{}".format(str(allfeatureload1_nosuffix)))
    ModelTrainAndTest(allTrainedPD, allTestPD, spath=spath, selectedFeature=allfeatureload1_nosuffix, modelpath="Classifiers/saved_model/tmp_load1_nosuffix")

