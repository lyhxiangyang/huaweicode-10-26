from utils.ParsingJson import covertCSVToJsonDict, saveDictToJson

if __name__ == "__main__":
    # ============================================================================================= 输入数据定义
    # 先将所有的server文件和process文件进行指定
    # 其中单个server文件我默认是连续的
    predictdirpath = R"DATA\L3l2数据集合"
    # 指定正常server和process文件路径
    normaldirpath = R"DATA\L3l2数据集合正常值"
    spath = "tmp/jsonfile"
    # 是否有存在faultFlag isExistFaultFlag = True
    # 核心数据 如果isManuallyspecifyCoreList==True那么就专门使用我手工指定的数据，如果==False，那么我使用的数据就是从process文件中推理出来的结果
    server_feature = ["mem_used", "pgfree", "freq"]
    # 需要对process数据进行处理的指标, cpu数据要在数据部分添加, 在后面，会往这个列表中添加一个cpu数据
    process_feature = ["usr_cpu", "kernel_cpu"]
    # 需要对l2数据进行处理的指标，
    l2_feature = ["cpu_power", "power", "cabinet_power",
                  "fan1_speed", "FAN1_R_Speed",
                  "fan2_speed", "FAN2_R_Speed",
                  "fan3_speed", "FAN3_R_Speed",
                  "fan4_speed", "FAN4_R_Speed",
                  'FAN5_F_Speed', "FAN5_R_Speed",
                  'FAN6_F_Speed', "FAN6_R_Speed",
                  'FAN7_F_Speed', "FAN7_R_Speed",
                  "cpu1_core_rem", "cpu2_core_rem", "cpu3_core_rem", "cpu4_core_rem",
                  "cpu1_mem_temp", "cpu2_mem_temp", "cpu3_mem_temp", "cpu4_mem_temp",
                  ]
    # 需要对网络数据进行处理的指标
    network_feature = ["tx_packets_phy", "rx_packets_phy"]
    normalMeanDict = {
        "server": {
            "mem_used": 48935662250,
            "pgfree": 351644,
            "freq": 3323,
        },
        "compute": {
            "cpu_power": 789.5757575757576,
            "power": 1157.8181818181818,
            "cabinet_power": 14168.727272727272,
            "fan1_speed": 6486.818181818182,
            "FAN1_R_Speed": 5955.0,
            "fan2_speed": 6493.636363636364,
            "FAN2_R_Speed": 5931.818181818182,
            "fan3_speed": 6488.181818181818,
            "FAN3_R_Speed": 5972.727272727273,
            "fan4_speed": 6416.090909090909,
            "FAN4_R_Speed": 6565.818181818182,
            "FAN5_F_Speed": 6499.090909090909,
            "FAN5_R_Speed": 5938.636363636364,
            "FAN6_F_Speed": 6488.181818181818,
            "FAN6_R_Speed": 5941.363636363636,
            "FAN7_F_Speed": 6100.0,
            "FAN7_R_Speed": 6591.69696969697,
            "cpu1_core_rem": 41.36363636363637,
            "cpu2_core_rem": 41.95454545454545,
            "cpu3_core_rem": 45.21212121212121,
            "cpu4_core_rem": 48.57575757575758,
            "cpu1_mem_temp": 51.72727272727273,
            "cpu2_mem_temp": 51.803030303030305,
            "cpu3_mem_temp": 47.59090909090909,
            "cpu4_mem_temp": 47.0,
        },
        "process": {
            "cpu": 60,
        },
        "nic": {
            "tx_packets_phy": 741384.6,
            "rx_packets_phy": 732001,
        }
    }



    # ========================================================= 进行读取
    jsonDict = covertCSVToJsonDict(predictdir=predictdirpath, normaldir=normaldirpath, normalMeanDict=normalMeanDict)
    saveDictToJson(jsonDict, spath=spath, filename="alljson.json")

