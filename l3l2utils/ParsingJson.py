"""
对json格式的转化
"""
import json
import os.path
from typing import Dict, List

import pandas as pd

from utils.DataFrameOperation import mergeDataFrames
from utils.auto_forecast import getServer_Process_l2_NetworkList

"""
将dataFrame转化为
{
0: {time: **, pid: **}
}
"""


def convertDataFrameToDict(df: pd.DataFrame) -> Dict:
    cdfDict = df.to_dict(orient='index')
    return cdfDict


"""
传进来的Dict类型是
{
0: {time: **, pid: **}
}
"""


def convertDictToDataFrame(dfdict: Dict) -> pd.DataFrame:
    df = pd.DataFrame(data=dfdict)
    return df


"""
将process、server、l2、network中的数据全部读取进来
filepath是存储network、server等目录的目录
"""


def covertCSVToJsonDict(predictdir: str, normaldir: str, server_feature=None,
                        process_feature=None,
                        l2_feature=None,
                        network_feature=None,
                        isExistFlag: bool = True,
                        jobid: int = 16,
                        type: str = 'L3',
                        requestdataType: str = 'wrf'):
    serverpds, processpds, l2pds, networkpds = getServer_Process_l2_NetworkList(predictdir,
                                                                                server_feature=server_feature,
                                                                                process_feature=process_feature,
                                                                                l2_feature=l2_feature,
                                                                                network_feature=network_feature,
                                                                                isExistFlag=isExistFlag)
    serverallpd, _ = mergeDataFrames(serverpds)
    processallpd, _ = mergeDataFrames(processpds)
    l2allpd, _ = mergeDataFrames(l2pds)
    networkallpd, _ = mergeDataFrames(networkpds)
    jsonDict = {}
    jsonDict["JobID"] = jobid
    jsonDict["Type"] = type
    jsonDict["RequestData"] = {}
    jsonDict["RequestData"]["type"] = requestdataType
    jsonDict["RequestData"]["data"] = {}
    jsonDict["RequestData"]["data"]["server"] = convertDataFrameToDict(serverallpd)
    jsonDict["RequestData"]["data"]["process"] = convertDataFrameToDict(processallpd)
    jsonDict["RequestData"]["data"]["network"] = convertDataFrameToDict(serverallpd)
    jsonDict["RequestData"]["data"]["l2"] = convertDataFrameToDict(serverallpd)
    return jsonDict


"""
将json保存与读取
"""


def saveDictToJson(sdict: Dict, spath: str, filename: str):
    if not os.path.exists(spath):
        os.makedirs(spath)
    pathfilename = os.path.join(spath, filename)
    with open(pathfilename, "w") as f:
        json.dump(sdict, f)


def readJsonToDict(spath: str, filename: str):
    pathfilename = os.path.join(spath, filename)
    with open(pathfilename, "r", encoding='utf-8') as f:
        jsonDict = json.load(f)
    return jsonDict


"""
从读取到的json文件得到server数据
"""


def getServerPdFromJsonDict(sdict: Dict) -> List[pd.DataFrame]:
    serverDict = sdict["RequestData"]["data"]["server"]
    serpd = pd.DataFrame(data=serverDict).T
    return [serpd]


"""
从读取到的json文件中得到process数据
"""


def getProcessPdFromJsonDict(sdict: Dict) -> List[pd.DataFrame]:
    processDict = sdict["RequestData"]["data"]["process"]
    processpd = pd.DataFrame(data=processDict).T
    return [processpd]


"""
从读取到的json文件中得到network数据
"""


def getNetworkPdFromJsonDict(sdict: Dict) -> List[pd.DataFrame]:
    networkDict = sdict["RequestData"]["data"]["network"]
    networkpd = pd.DataFrame(data=networkDict).T
    return [networkpd]


"""
从读取到的json文件中得到l2数据
"""


def getL2PdFromJsonDict(sdict: Dict) -> List[pd.DataFrame]:
    l2Dict = sdict["RequestData"]["data"]["l2"]
    l2pd = pd.DataFrame(data=l2Dict).T
    return [l2pd]


"""
得到正常server数据的平均值
todo
"""

"""
函数功能：从已经存在的detection中获得现有的平均值
函数参数:
detectionJson: 输入数据
classname: 别分代表l2 server process network
"""


def getMeanFromExistMean(detectionJson: Dict, classname: str, featuresname: str):
    classFeatureMeanDict = detectionJson["RequestData"]["normalDataMean"][classname]
    meanValue = None
    if featuresname in classFeatureMeanDict.keys():
        meanValue = Dict["RequestData"]["normalDataMean"][classname][featuresname]
    return meanValue


def getMeanFromDataFrom(detectionJson: Dict, classname: str, featuresnames: List[str], datanumber: int = 10):
    dataDict = detectionJson["RequestData"]["data"][classname]
    classdatanumber = min(datanumber, len(dataDict))
    dataDict = dict(list(dataDict.items())[:classdatanumber])
    dataPd = pd.DataFrame(data=dataDict[:classdatanumber])
    return dataPd[featuresnames].mean()


def getNormalServerMean(detectionJson: Dict, features: List[str], datanumber: int = 10) -> pd.Series:
    meanSeries = getMeanFromDataFrom(detectionJson, "server", features, datanumber)
    for ifeaturename in features:
        featureVaule = getMeanFromExistMean(detectionJson, "server", ifeaturename)
        if featureVaule is not None:
            meanSeries[ifeaturename] = featureVaule
    return meanSeries


def getNormalProcessMean(detectionJson: Dict, features: List[str], datanumber: int = 10) -> pd.Series:
    meanSeries = getMeanFromDataFrom(detectionJson, "process", features, datanumber)
    for ifeaturename in features:
        featureVaule = getMeanFromExistMean(detectionJson, "process", ifeaturename)
        if featureVaule is not None:
            meanSeries[ifeaturename] = featureVaule
    return meanSeries


def getNormalL2Mean(detectionJson: Dict, features: List[str], datanumber: int = 10) -> pd.Series:
    meanSeries = getMeanFromDataFrom(detectionJson, "l2", features, datanumber)
    for ifeaturename in features:
        featureVaule = getMeanFromExistMean(detectionJson, "l2", ifeaturename)
        if featureVaule is not None:
            meanSeries[ifeaturename] = featureVaule
    return meanSeries


def getNormalNetworkMean(detectionJson: Dict, features: List[str], datanumber: int = 10) -> pd.Series:
    meanSeries = getMeanFromDataFrom(detectionJson, "network", features, datanumber)
    for ifeaturename in features:
        featureVaule = getMeanFromExistMean(detectionJson, "network", ifeaturename)
        if featureVaule is not None:
            meanSeries[ifeaturename] = featureVaule
    return meanSeries
