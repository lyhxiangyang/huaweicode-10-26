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
    server_feature = ["used", "pgfree", "freq"]
    # 需要对process数据进行处理的指标, cpu数据要在数据部分添加, 在后面，会往这个列表中添加一个cpu数据
    process_feature = ["user", "system"]
    # 需要对l2数据进行处理的指标，
    l2_feature = ["CPU_Powewr", "Power", "Cabinet_Power",
                  'FAN1_F_Speed', "FAN1_R_Speed",
                  'FAN2_F_Speed', "FAN2_R_Speed",
                  'FAN3_F_Speed', "FAN3_R_Speed",
                  'FAN4_F_Speed', "FAN4_R_Speed",
                  'FAN5_F_Speed', "FAN5_R_Speed",
                  'FAN6_F_Speed', "FAN6_R_Speed",
                  'FAN7_F_Speed', "FAN7_R_Speed",
                  'CPU1_Core_Rem', 'CPU2_Core_Rem', 'CPU3_Core_Rem', 'CPU4_Core_Rem',
                  'CPU1_MEM_Temp', 'CPU2_MEM_Temp', 'CPU3_MEM_Temp', 'CPU4_MEM_Temp',
                  ]
    # 需要对网络数据进行处理的指标
    network_feature = ["tx_packets_phy", "rx_packets_phy"]
    normalMeanDict = {
        "server": {
            "used": 48935662250,
            "pgfree": 351644,
            "freq": 3323,
        },
        "l2": {
            "CPU_Powewr": 789.5757575757576,
            "Power": 1157.8181818181818,
            "Cabinet_Power": 14168.727272727272,
            "FAN1_F_Speed": 6486.818181818182,
            "FAN1_R_Speed": 5955.0,
            "FAN2_F_Speed": 6493.636363636364,
            "FAN2_R_Speed": 5931.818181818182,
            "FAN3_F_Speed": 6488.181818181818,
            "FAN3_R_Speed": 5972.727272727273,
            "FAN4_F_Speed": 6416.090909090909,
            "FAN4_R_Speed": 6565.818181818182,
            "FAN5_F_Speed": 6499.090909090909,
            "FAN5_R_Speed": 5938.636363636364,
            "FAN6_F_Speed": 6488.181818181818,
            "FAN6_R_Speed": 5941.363636363636,
            "FAN7_F_Speed": 6100.0,
            "FAN7_R_Speed": 6591.69696969697,
            "CPU1_Core_Rem": 41.36363636363637,
            "CPU2_Core_Rem": 41.95454545454545,
            "CPU3_Core_Rem": 45.21212121212121,
            "CPU4_Core_Rem": 48.57575757575758,
            "CPU1_MEM_Temp": 51.72727272727273,
            "CPU2_MEM_Temp": 51.803030303030305,
            "CPU3_MEM_Temp": 47.59090909090909,
            "CPU4_MEM_Temp": 47.0,
        },
        "process": {
            "cpu": 60,
        },
        "network": {
            "tx_packets_phy": 741384.6,
            "rx_packets_phy": 732001,
        }
    }



    # ========================================================= 进行读取
    jsonDict = covertCSVToJsonDict(predictdir=predictdirpath, normaldir=normaldirpath, normalMeanDict=normalMeanDict)
    saveDictToJson(jsonDict, spath=spath, filename="alljson.json")

