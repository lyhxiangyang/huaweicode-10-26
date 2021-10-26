"""
这个文件的本意是包含对DataFrame的各种操作
如：
1. 将多个相同类型的DataFrame合并成一个
2. 将一个大的DataFrame根据某个特征值分割成多个
3. 按照列名对DataFrame的每一列进行调换顺序等


"""
from collections import defaultdict
from typing import List, Dict, Tuple, Union
from utils.DefineData import *
import pandas as pd

"""
函数功能：合并多个DataFrame 返回值是一个合并之后的DataFrame 上下合并
函数参数：预期是一个含有DataFrame的列表 
返回值： 包含各个DataFrame的一个大的DataFrame , 第二个参数表示是否运行出错 True 代表出错  False代表没错
"""


def mergeDataFrames(lpds: List[pd.DataFrame]) -> (pd.DataFrame, bool):
    # 如果说传进来的DataFrame头部不一样 就报错
    if not judgeSameFrames(lpds):
        return None, False
    # 将列表中的数据全都合并
    dfres = pd.concat(lpds, ignore_index=True)
    return dfres, False


"""
函数功能： 判断多个DataFrame是否含有相同的列表
函数参数：预期是一个含有DataFrame的列表 
函数参数：True 含有相同的列名     False 不含有相同的列名
"""


def judgeSameFrames(lpds: List[pd.DataFrame]) -> bool:
    lcolumns = [list(i.columns.array) for i in lpds]

    # 长度为0的时候可以返回True
    if len(lpds) == 0 or len(lpds) == 1:
        return True
    for i in range(1, len(lcolumns)):
        if lcolumns[i] != lcolumns[0]:
            return False
    return True


# 将一个DataFrame的FAULT_FLAG重值为ff
"""
函数功能： 将DataFrame中的FAULT_FLAG设置为ff
"""


def setPDfaultFlag(df: pd.DataFrame, ff: int) -> pd.DataFrame:
    if FAULT_FLAG in df.columns.array:
        df = df.drop(FAULT_FLAG, axis=1)
    lengthpd = len(df)
    ffdict = {FAULT_FLAG: [ff] * lengthpd}
    tpd = pd.DataFrame(data=ffdict)
    tpd = pd.concat([df, tpd], axis=1)
    return tpd


"""
函数功能： 将DataFrame中按照FAULT_FLAG进行分类生成
函数参数： 传入一个Server的信息 包含Fault_Flag参数
函数返回值： Dict 第一个是错误码 对应是其DataFrame结构
"""


def divedeDataFrameByFaultFlag1(df: pd.DataFrame, isMerged : bool = True) -> (Dict[int, pd.DataFrame], bool):
    # 先判断是否存在Fault_FLag参数，不存在就报错
    if FAULT_FLAG not in list(df.columns.array):
        return None, True

    # 对Fault_Flag这一行进行去重, 并将错误
    sFault_Flag_Colums = sorted(list(set(df[FAULT_FLAG])))

    # 重复n个空的DataFrame， 方便使用zip直接生成一个Dict结构的数据结构
    # repeatEmptyDataFrames = [pd.DataFrame(columns=df.columns.array) for i in range(0, len(sFault_Flag_Colums))]
    # resDict = dict(zip(sFault_Flag_Colums, repeatEmptyDataFrames))
    resDict: Dict = {}

    # 遍历DataFrame根据 Fault_Flag这一行来分开
    for ifault in sFault_Flag_Colums:
        tfault = ifault
        if isMerged:
            tfault = ifault // 10
        ipd = df.loc[df[FAULT_FLAG] == ifault].copy()
        ipd.reset_index(drop=True, inplace=True)
        ipd = setPDfaultFlag(ipd, tfault)
        if tfault not in resDict.keys():
            resDict[tfault] = ipd
            continue
        tipd, err = mergeDataFrames([resDict[tfault], ipd])
        resDict[tfault] = tipd

    return resDict, False


"""
函数功能： 将DataFrame中按照FAULT_FLAG进行分类生成
函数参数： 传入一个Server的信息 包含Fault_Flag参数
函数返回值： Dict 第一个是错误码 对应是其DataFrame结构
"""


def divedeDataFrameByFaultFlag(df: pd.DataFrame) -> (Dict[int, pd.DataFrame], bool):
    # 先判断是否存在Fault_FLag参数，不存在就报错
    if FAULT_FLAG not in list(df.columns.array):
        return None, True

    # 对Fault_Flag这一行进行去重, 并将错误
    sFault_Flag_Colums = sorted(list(set(df[FAULT_FLAG])))

    # 重复n个空的DataFrame， 方便使用zip直接生成一个Dict结构的数据结构
    # repeatEmptyDataFrames = [pd.DataFrame(columns=df.columns.array) for i in range(0, len(sFault_Flag_Colums))]
    # resDict = dict(zip(sFault_Flag_Colums, repeatEmptyDataFrames))
    resDict: Dict = {}

    # 遍历DataFrame根据 Fault_Flag这一行来分开
    for i in range(0, len(df)):
        # 第i行的获取
        df_iline = df.iloc[i].copy()
        # 错误码
        iFault_Flag_Number = (df_iline[FAULT_FLAG] // 10) * 10
        # 修改FAULT_FLAG值
        df_iline[FAULT_FLAG] = iFault_Flag_Number
        if iFault_Flag_Number not in resDict.keys():
            resDict[iFault_Flag_Number] = pd.DataFrame(columns=df.columns.array)
        # 在对应字典中添加一行, 忽略index 逐步增加
        resDict[iFault_Flag_Number] = resDict[iFault_Flag_Number].append(df_iline, ignore_index=True)

    # =================================================
    # 显示DEBUG的信息
    if DEBUG:
        print("divedeDataFrameByFaultFlag".center(40, "*"))
        for k, kpd in resDict.items():
            print(str(k).center(10, "="))
            print(kpd)
        print("end".center(40, "*"))
    # =================================================

    # 返回
    return resDict, False


'''
# - 功能介绍
# 将dataFrame中的一个名字为lable的列名字移动到最前面
# - 参数介绍
# 1. dataFrame是要移动的表格信息
# 2. 要修改的标签，如果没有这个标签，就不移动
# - 返回值介绍
# 放回第一个参数这个表格
'''


def PushLabelToFirst(dataFrame: pd.DataFrame, label: str) -> pd.DataFrame:
    columnsList = list(dataFrame.columns)
    if label not in columnsList:
        return dataFrame
    columnsList.insert(0, columnsList.pop(columnsList.index(label)))
    dataFrame = dataFrame[columnsList]
    return dataFrame


'''
# - 功能介绍
# 将dataFrame中的一个名字为lable的列名字移动到最后面
# - 参数介绍
# 1. dataFrame是要移动的表格信息
# 2. 要修改的标签，如果没有这个标签，就不移动
# - 返回值介绍
# 放回第一个参数这个表格
'''


def PushLabelToEnd(dataFrame: pd.DataFrame, label: str) -> pd.DataFrame:
    columnsList = list(dataFrame.columns)
    if label not in columnsList:
        return dataFrame
    columnsList.append(columnsList.pop(columnsList.index(label)))
    dataFrame = dataFrame[columnsList]
    return dataFrame


'''
-  功能介绍：
   用来一个DataFrame的列按照列名重新排序，使其列按照一定顺序排列
-  参数介绍：
   1. dataFrame是我们要排序的表
   2. reverse=False表示列名是按照字符串从小到大排列，True表示从大到小
-  返回值介绍：
   1. 表示排序好得到的DataFrame
'''


def SortLabels(dataFrame: pd.DataFrame, reverse=False) -> pd.DataFrame:
    columnsList = list(dataFrame.columns)
    columnsList.sort(reverse=reverse)
    dataFrame = dataFrame[columnsList]
    return dataFrame


"""
-   功能介绍：
    用来判断某个PD里面是否有控制，用于检测bug
-   参数介绍：
    
"""


def isEmptyInDataFrame(targetDF: pd.DataFrame) -> bool:
    tSeries: pd.Series = targetDF.isnull().any()
    isHaveBool = [i for i in tSeries if i]
    if len(isHaveBool) == 0:
        return False
    return True


"""
-   功能介绍：
    将一个DataFrame中的所有行中对应的特征值都都剪去第一行，除去time和flagFault
-   返回值一个DataFrame和一个是否有错误的bool类型

-   重点：传入的参数必须是index=[0, 1, 2]
    传入前可以通过reset_index(drop=True, inplace=True)
"""


def subtractFirstLineFromDataFrame(df: pd.DataFrame, columns: List) -> Union[
    Tuple[None, bool], Tuple[pd.DataFrame, bool]]:
    if len(df) == 0:
        return None, True
    # https://www.jianshu.com/p/72274ccb647a
    # 注意会出现这种警告
    for iline in range(1, len(df)):
        df.loc[iline, columns] = df.loc[iline, columns] - df.loc[0, columns]
    df.loc[0, columns] = df.loc[0, columns] - df.loc[0, columns]
    return df, False

"""
-   功能介绍：
    将一个DataFrame中的所有行中对应的特征值都都剪去第一行，除去time和flagFault
-   返回值一个DataFrame和一个是否有错误的bool类型

-   重点：传入的参数必须是index=[0, 1, 2]
    传入前可以通过reset_index(drop=True, inplace=True)
"""


def subtractLastLineFromDataFrame(df: pd.DataFrame, columns: List) -> Union[
    Tuple[None, bool], Tuple[pd.DataFrame, bool]]:
    if len(df) <= 1:
        return None, True
    # https://www.jianshu.com/p/72274ccb647a
    # 注意会出现这种警告
    for iline in range(len(df) - 1, 0, -1):
        df.loc[iline, columns] = df.loc[iline, columns] - df.loc[iline - 1, columns]

    df.loc[0, columns] = df.loc[1, columns]
    return df, False

# 合并的两个类型是fault-DataFrame
def mergeTwoDF(dic1: Dict[int, pd.DataFrame], dic2: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
    allfaulty = list(dic1.keys())
    allfaulty.extend(list(dic2.keys()))
    allfauly = list(set(allfaulty))
    resDict = {}
    for ifaulty in allfauly:
        tpd: pd.DataFrame = pd.DataFrame()
        if ifaulty in dic1 and ifaulty in dic2:
            tpd = pd.concat([dic1[ifaulty], dic2[ifaulty]], ignore_index=True)
        elif ifaulty in dic1:
            tpd = dic1[ifaulty]
        elif ifaulty in dic2:
            tpd = dic2[ifaulty]
        resDict[ifaulty] = tpd
    return resDict
