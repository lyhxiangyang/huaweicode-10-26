

# 将server时间显示的内存两减去process的内存，得到一个外部内存使用量的变化
if __name__ == "__main__":

    #  ======================================================== 确定所有的配置信息
    processFeatures = ["pmem"] # 要计算和的所有指标
    processAccumulateFeatures = ["pmem"] # 上面指标中，是累计数据的指标名字，用来处理
    processpdfile = [] # processpd的路径
    serverpdfile = [] # serverpd的路径

    # ======================================================== 确定所有的配置信息
    print("将所有的process文件进行合并累计求和".center(40, "*"))

