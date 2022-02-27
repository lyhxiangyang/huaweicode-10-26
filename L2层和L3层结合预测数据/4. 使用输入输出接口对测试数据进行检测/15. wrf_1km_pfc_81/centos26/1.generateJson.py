import os.path

from hpc.l3l2utils.ParsingJson import covertCSVToJsonDict, saveDictToJson

if __name__ == "__main__":
    # ============================================================================================= 输入数据定义
    # 先将所有的server文件和process文件进行指定
    # 其中单个server文件我默认是连续的
    predictdirpath = R"DATA/2022-01-14新的测试数据/15.wrf_1km_multi_pfc_81/centos26"
    spath = os.path.join(predictdirpath, "jsonfile") # 将结果和文件生成到一起
    jsonfilename = "alljson.json"
    normalMeanDict = {
        "server": {
            # "mem_used": 48935662250,
            # "pgfree": 351644,
            # "freq": 3323,
        },
        "compute": {
            # "cpu_power": 789.5757575757576,
            # "power": 1157.8181818181818,
            # "cabinet_power": 14168.727272727272,
            # "fan1_speed": 6486.818181818182,
            # "FAN1_R_Speed": 5955.0,
            # "fan2_speed": 6493.636363636364,
            # "FAN2_R_Speed": 5931.818181818182,
            # "fan3_speed": 6488.181818181818,
            # "FAN3_R_Speed": 5972.727272727273,
            # "fan4_speed": 6416.090909090909,
            # "FAN4_R_Speed": 6565.818181818182,
            # "FAN5_F_Speed": 6499.090909090909,
            # "FAN5_R_Speed": 5938.636363636364,
            # "FAN6_F_Speed": 6488.181818181818,
            # "FAN6_R_Speed": 5941.363636363636,
            # "FAN7_F_Speed": 6100.0,
            # "FAN7_R_Speed": 6591.69696969697,
            # "cpu1_core_rem": 41.36363636363637,
            # "cpu2_core_rem": 41.95454545454545,
            # "cpu3_core_rem": 45.21212121212121,
            # "cpu4_core_rem": 48.57575757575758,
            # "cpu1_mem_temp": 51.72727272727273,
            # "cpu2_mem_temp": 51.803030303030305,
            # "cpu3_mem_temp": 47.59090909090909,
            # "cpu4_mem_temp": 47.0,
        },
        "process": {
            "cpu": 60,
        },
        "nic": {
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

