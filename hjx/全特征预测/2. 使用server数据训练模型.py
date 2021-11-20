from typing import List

import pandas as pd

from Classifiers.TrainToTest import ModelTrainAndTest
from utils.DataFrameOperation import mergeDataFrames, sortByAbsValue
from utils.DefineData import FAULT_FLAG

trainNormalDataPath = [
    R"tmp/11-19-tData/测试数据-E5-3km/server/4. server上数据异常信息集合/0.csv", #hjx 1
]

trainAbnormalDataPath = [
    (60, R"tmp/11-19-tData/测试数据-E5-3km/server/4. server上数据异常信息集合/61.csv"), #hjx2
    (60, R"tmp/11-19-tData/测试数据-E5-3km/server/4. server上数据异常信息集合/62.csv"),
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
    spath = "tmp/E5多机预测Local-标准化特征提取-合并错误" #hjx2
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


    #==================================================================将正常的训练中截取一部分和异常数据等长的数据
    # 获得测试数据的总长度
    # lenAbnormalData = sum([len(ipd) for ipd in trainAbnormalList]) // 2
    # tmplist = []
    # for ipd in trainNormalList:
    #     ipd = sortByAbsValue(ipd, "cpu_mean", 60)
    #     ipd = ipd.loc[0: lenAbnormalData, :]
    #     tmplist.append(ipd)
    # trainNormalList = tmplist
    #==================================================================en

    #==================================================================将所有的训练数据进行合并
    allTrainedPD, err = mergeDataFrames(trainNormalList + trainAbnormalList)
    if err:
        print("训练数据合并失败")
        exit(1)

    # 深度为2， 特征名为 pgfree_mean
    # max_depth = 3 # hjx3
    # allfeatureload1_nosuffix = ["used_mean"] # hjx3

    # 获得需要训练的特征
    max_depth = 5 # hjx4
    allfeatureload1_nosuffix = get_List_pre_nosuffix(list(allTrainedPD.columns.array), prefix="used_", suffix="_diff") # hjx4


    print("选择的特征：{}".format(str(allfeatureload1_nosuffix)))
    ModelTrainAndTest(allTrainedPD, None, spath=spath, selectedFeature=allfeatureload1_nosuffix,
                      modelpath="Classifiers/saved_model/tmp_load1_nosuffix", #hjx6
                      testAgain=False, maxdepth=max_depth)

