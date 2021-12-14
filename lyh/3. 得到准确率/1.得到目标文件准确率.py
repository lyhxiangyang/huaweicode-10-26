import pandas as pd

from utils.DataFrameOperation import mergeDataFrames
from utils.auto_forecast import removeAllHeadTail, getAccuracy

if __name__ == "__main__":
    accuracyFilesPath = [
        R"tmp/总过程分析/测试数据-E5-1KM/6. 最终预测结果/预测结果.csv",
        R"tmp/总过程分析/测试数据-E5-3KM/6. 最终预测结果/预测结果.csv",
        R"tmp/总过程分析/测试数据-E5-8KM/6. 最终预测结果/预测结果.csv",
        R"tmp/总过程分析/测试数据-E5-RST/6. 最终预测结果/预测结果.csv",
        R"tmp/总过程分析/测试数据-Local-1km/6. 最终预测结果/预测结果.csv",
        R"tmp/总过程分析/测试数据-Local-3km/6. 最终预测结果/预测结果.csv",
        R"tmp/总过程分析/测试数据-Local-9km/6. 最终预测结果/预测结果.csv",
    ]
    accpds = [pd.read_csv(ipath) for ipath in accuracyFilesPath]
    # 去除每个点的首尾连个字符
    accpds = [ removeAllHeadTail(ipd, windowsize=3) for ipd in accpds ]
    # 合并
    allaccpd, _ = mergeDataFrames(accpds)
    faultFlagName = list(allaccpd["faultFlag"])
    preFlagName = list(allaccpd["preFlag"])
    acc = getAccuracy(realflags=faultFlagName, preflags=preFlagName, excludeflags=[0])
    print("wrf所有的准确率：{:.2%}".format(acc))

    accuracyFilesPath = [
        R"tmp/总过程分析/Grapes/测试数据-E5/6. 最终预测结果/预测结果.csv",
        R"tmp/总过程分析/Grapes/测试数据-Local/6. 最终预测结果/预测结果.csv",
    ]
    accpds = [pd.read_csv(ipath) for ipath in accuracyFilesPath]
    # 去除每个点的首尾连个字符
    accpds = [removeAllHeadTail(ipd, windowsize=3) for ipd in accpds]
    # 合并
    allaccpd, _ = mergeDataFrames(accpds)
    faultFlagName = list(allaccpd["faultFlag"])
    preFlagName = list(allaccpd["preFlag"])
    acc = getAccuracy(realflags=faultFlagName, preflags=preFlagName)
    print("Grape所有的准确率：{:.2%}".format(acc))



