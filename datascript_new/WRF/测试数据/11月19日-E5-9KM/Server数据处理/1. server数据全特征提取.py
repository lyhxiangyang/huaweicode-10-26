import os

from utils.DataFrameOperation import mergeDataFrames
from utils.DataScripts import getDFmean
from utils.DefineData import TIME_COLUMN_NAME, FAULT_FLAG
from utils.FileSaveRead import saveDFListToFiles, saveFaultyDict
from utils.auto_forecast import getfilespath, getfilepd, differenceServer, \
    standardLists, changeTimeTo_pdlists, serverpdsList, allMistakesOnExtractingOneCore

if __name__ == "__main__":
    # ============================================================================================= 输入数据定义
    # 先将所有的server文件和process文件进行指定
    # 其中单个server文件我默认是连续的
    predictdirpath = R"C:\Users\lWX1084330\Desktop\正常和异常数据\测试数据-E5-9km-异常数据"
    predictserverfiles = getfilespath(os.path.join(predictdirpath, "server"))
    # 指定正常server和process文件路径
    normaldirpath = R"C:\Users\lWX1084330\Desktop\正常和异常数据\E5-3km-正常数据"
    normalserverfiles = getfilespath(os.path.join(normaldirpath, "server"))

    spath = "tmp/11-19-tData/测试数据-E5-9km/server"

    # 需要对server数据进行处理的指标
    server_feature = ["used", "pgfree"]
    server_accumulate_feature = ["pgfree"]

    # 在处理时间格式的时候使用，都被转化为'%Y-%m-%d %H:%M:00' 在这里默认所有的进程数据是同一种时间格式，
    server_time_format = '%Y/%m/%d %H:%M'

    # ============================================================================================= 先将正常数据和预测数据的指标从磁盘中加载到内存中
    print("将数据从文件中读取".center(40, "*"))
    normalserverpds = []
    predictserverpds = []

    # 加入time faultFlag特征值
    time_server_feature = server_feature.copy()
    # 加入时间是为了process和server的对应， 加入pid 是为了进行分类。加入CPU是为了预测哪个CPU出现了异常
    time_server_feature.extend([TIME_COLUMN_NAME])
    # flagFault要视情况而定
    time_server_feature.append(FAULT_FLAG)


    # 正常服务数据
    for ifile in normalserverfiles:
        tpd = getfilepd(ifile)
        tpd = tpd.loc[:, time_server_feature]
        normalserverpds.append(tpd)

    # 预测服务数据
    for ifile in predictserverfiles:
        tpd = getfilepd(ifile)
        tpd = tpd.loc[:, time_server_feature]
        predictserverpds.append(tpd)
    # ============================================================================================= 对读取到的数据进行差分，并且将cpu添加到要提取的特征中
    print("对读取到的原始数据进行差分".format(40, "*"))
    # 对正常server进程数据进行差分处理之后，得到一些指标
    normalserverpds = differenceServer(normalserverpds, server_accumulate_feature)
    # 对异常server服务数据进行差分处理之后，得到一些指标
    predictserverpds = differenceServer(predictserverpds, server_accumulate_feature)

    # ============================================================================================= 先对正常数据的各个指标求平均值
    # 往进程指标中只使用"cpu"指标, 需要保证正常数据中的累计值都减去了

    print("先对正常数据的各个指标求平均值".center(40, "*"))
    allnormalserverpd, _ = mergeDataFrames(normalserverpds)
    # 得到正常数据的平均值
    normalserver_meanvalue = getDFmean(allnormalserverpd, server_feature)
    # 将这几个平均值进行保存
    tpath = os.path.join(spath, "1. 正常数据的平均值")
    if not os.path.exists(tpath):
        os.makedirs(tpath)
    normalserver_meanvalue.to_csv(os.path.join(tpath, "meanvalue_server.csv"))

    # ============================================================================================= 对要预测的数据进行标准化处理
    # 标准化process 和 server数据， 对于process数据，先将cpu想加在一起，然后在求平均值。
    print("标准化要预测的process和server数据".center(40, "*"))
    standard_server_pds = standardLists(pds=predictserverpds, standardFeatures=server_feature,
                                        meanValue=normalserver_meanvalue, standardValue=100)
    # 对标准化结果进行存储
    tpath = os.path.join(spath, "2. 标准化数据存储")
    saveDFListToFiles(os.path.join(tpath, "server_standard"), standard_server_pds)
    # ============================================================================================= 对process数据和server数据进行秒数的处理，将秒数去掉
    standard_server_pds = changeTimeTo_pdlists(standard_server_pds, server_time_format)
    # ============================================================================================= 对process数据和server数据进行特征提取
    print("对server数据进行特征处理".center(40, "*"))
    tpath = os.path.join(spath, "3. server特征提取数据")
    extraction_server_pds = serverpdsList(standard_server_pds, extractFeatures=server_feature,
                                          windowsSize=3, spath=tpath)
    # ============================================================================================= 对server数据进行错误的提取
    print("对server数据进行错误提取".center(40, "*"))
    allserverpds, _ = mergeDataFrames(extraction_server_pds)
    # 把这个server文件当作一个核上的数据，可以进行提取
    faultpdDict = allMistakesOnExtractingOneCore(allserverpds, windowsize=3)
    tpath = os.path.join(spath, "4. server上数据异常信息集合")
    saveFaultyDict(tpath, faultpdDict)

