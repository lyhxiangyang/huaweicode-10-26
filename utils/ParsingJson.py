"""
对json格式的转化
"""
import json
import os.path
from typing import Dict

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


def covertCSVToJsonDict(dirpath: str, server_feature,
                        process_feature,
                        l2_feature,
                        network_feature,
                        isExistFlag: bool = True,
                        jobid: int = 16,
                        type: str = 'L3',
                        requestdataType: str = 'wrf'):
    serverpds, processpds, l2pds, networkpds = getServer_Process_l2_NetworkList(dirpath, server_feature=server_feature,
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
将json保存
"""


def saveDictToJson(sdict: Dict, spath: str, filename: str):
    pathfilename = os.path.join(spath, filename)
    with open(pathfilename, "w") as f:
        json.dump(sdict, f)
def readJsonToDict(spath: str, filename: str):
    pathfilename = os.path.join(spath, filename)
    jsonDict = {}
    with open(pathfilename, "r") as f:
        jsonDict = json.load(f)
    return jsonDict
