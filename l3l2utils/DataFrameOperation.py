from typing import List

import pandas as pd

from l3l2utils.DefineData import FAULT_FLAG, TIME_COLUMN_NAME

"""
函数功能：合并多个DataFrame 返回值是一个合并之后的DataFrame 上下合并
函数参数：预期是一个含有DataFrame的列表
返回值： 包含各个DataFrame的一个大的DataFrame , 第二个参数表示是否运行出错 True 代表出错  False代表没错
"""


def mergeDataFrames(lpds: List[pd.DataFrame]):
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
            lleft = set(lcolumns[i])
            lright = set(lcolumns[0])
            if lleft != lright:
                a = lleft & lright
                b = lleft - a
                c = lright - a
                return False
        return True

    # 如果说传进来的DataFrame头部不一样 就报错
    if not judgeSameFrames(lpds):
        print("mergeDataFrames函数-合并失误")
        exit(1)
    # 将列表中的数据全都合并
    dfres = pd.concat(lpds, ignore_index=True)
    return dfres


"""
将两个DataFrame按照时间进行进行合并, 获得交集，
使用第一个df的faultFlag，第二个dataframe中的faultFlag不使用
"""


def mergeinnerTwoDataFrame(lpd: pd.DataFrame, rpd: pd.DataFrame, onfeaturename: str = TIME_COLUMN_NAME) -> pd.DataFrame:
    rpdcolumns = list(rpd.columns.array)
    rpdcolumns.remove(FAULT_FLAG)
    mergedf = pd.merge(left=lpd, right=rpd[rpdcolumns], left_on=onfeaturename, right_on=onfeaturename)
    return mergedf


"""
将多个预测结果合并起来
如果时间相同，那么就将结果合并起来
1. 保证索引不是time
time faultFlag  preFlag
"""


def mergeouterPredictResult(pds: List[pd.DataFrame]) -> pd.DataFrame:
    def fun_faultFlag(xlist: pd.Series):
        xlist = xlist.dropna()
        # 去重
        xlist = xlist[~xlist.duplicated()].reset_index(drop=True)
        assert len(xlist) != 0
        # return list(xlist)
        return int(xlist[0])

    # -1 代表着正常
    def fun_preFlag(xlist: pd.Series):
        xlist = xlist.dropna()
        assert len(xlist) != 0
        xlist = xlist[~xlist.duplicated()].reset_index(drop=True)  # 先去重
        xlist = list(map(int, xlist))
        # 如果有-1那么就代表process没有那一部分，
        if -1 in xlist:
            # 如果-1在这里，说明这个时间点process中没有wrf运行，那么我们去除
            # 把server预测的结果删掉 小于100的数值去掉，然后将0去掉
            xlist = [ilist for ilist in xlist if ilist > 100]
            xlist.append(0)
        xlist = sorted(list(xlist))
        # 如果有多个preFlag，那么就显示除了之外的所有preFlag
        if len(xlist) > 1:
            if 0 in xlist:
                xlist.remove(0)
            if -1 in xlist:
                xlist.remove(-1)
        return xlist

    # 需要将每个dataframe的索引设置为time
    timeindexpds = [ipd.set_index(TIME_COLUMN_NAME) for ipd in pds]
    mergepds = pd.concat(timeindexpds, join="outer", axis=1)  # 将会出现多个faultFlag和多个preFlag 将会按照time进行列的合并，会出现多行
    respd = pd.DataFrame(index=mergepds.index)  # 设置时间
    respd.loc[:, FAULT_FLAG] = mergepds[FAULT_FLAG].apply(fun_faultFlag, axis=1)
    respd.loc[:, "preFlag"] = mergepds["preFlag"].apply(fun_preFlag, axis=1)
    respd.sort_values(by="time", inplace=True)
    respd.reset_index(drop=False, inplace=True)
    return respd
