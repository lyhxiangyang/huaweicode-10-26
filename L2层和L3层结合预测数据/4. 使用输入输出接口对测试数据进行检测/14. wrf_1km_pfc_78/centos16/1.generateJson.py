import os.path

from hpc.l3l2utils.ParsingJson import covertCSVToJsonDict, saveDictToJson

if __name__ == "__main__":
    # ============================================================================================= 输入数据定义
    # 先将所有的server文件和process文件进行指定
    # 其中单个server文件我默认是连续的
    predictdirpath = R"DATA/2022-01-14新的测试数据/14.wrf_1km_multi_pfc_78/centos16"
    spath = os.path.join(predictdirpath, "jsonfile") # 将结果和文件生成到一起
    jsonfilename = "alljson.json"
    normalMeanDict = {
        "server": {
            # "used": 48935662250,
            # "pgfree": 351644,
            # "freq": 3323,
        },
        "l2": {
            # "CPU_Powewr": 789.5757575757576,
            # "Power": 1157.8181818181818,
            # "Cabinet_Power": 14168.727272727272,
            # "FAN1_F_Speed": 6486.818181818182,
            # "FAN1_R_Speed": 5955.0,
            # "FAN2_F_Speed": 6493.636363636364,
            # "FAN2_R_Speed": 5931.818181818182,
            # "FAN3_F_Speed": 6488.181818181818,
            # "FAN3_R_Speed": 5972.727272727273,
            # "FAN4_F_Speed": 6416.090909090909,
            # "FAN4_R_Speed": 6565.818181818182,
            # "FAN5_F_Speed": 6499.090909090909,
            # "FAN5_R_Speed": 5938.636363636364,
            # "FAN6_F_Speed": 6488.181818181818,
            # "FAN6_R_Speed": 5941.363636363636,
            # "FAN7_F_Speed": 6100.0,
            # "FAN7_R_Speed": 6591.69696969697,
            # "CPU1_Core_Rem": 41.36363636363637,
            # "CPU2_Core_Rem": 41.95454545454545,
            # "CPU3_Core_Rem": 45.21212121212121,
            # "CPU4_Core_Rem": 48.57575757575758,
            # "CPU1_MEM_Temp": 51.72727272727273,
            # "CPU2_MEM_Temp": 51.803030303030305,
            # "CPU3_MEM_Temp": 47.59090909090909,
            # "CPU4_MEM_Temp": 47.0,
        },
        "process": {
            "cpu": 60,
        },
        "network": {
            # "tx_packets_phy": 741384.6,
            # "rx_packets_phy": 732001,
        },
        "ping": {

        },
        "topdown": {

        }
    }



    # ========================================================= 进行读取
    jsonDict = covertCSVToJsonDict(predictdir=predictdirpath, normalMeanDict=normalMeanDict)
    saveDictToJson(jsonDict, spath=spath, filename=jsonfilename)

