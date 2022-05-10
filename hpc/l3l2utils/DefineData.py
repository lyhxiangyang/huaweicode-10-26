from enum import unique, IntEnum


@unique
class NodeState(IntEnum):
    pass


WINDOWS_SIZE = 5
FAULT_FLAG = "faultFlag"
TIME_COLUMN_NAME = "time"
CPU_FEATURE = "cpu_affinity"
PID_FEATURE = "pid"
TIME_INTERVAL = 60

PROCESS_CPUNAME = "cpu"
# 定义固定的文件名字
FDR = 0.01

# 存放模型的路径
SaveModelPath = 'Classifiers/saved_model'

# 需要排除的特征名字
EXCLUDE_FEATURE_NAME = ["time", FAULT_FLAG]

# 机器学习用到的常数
# 模型类型
MODEL_TYPE = ['decision_tree', 'random_forest', 'adaptive_boosting']

# 属于CPU和MEMORY异常类型的种类
CPU_ABNORMAL_TYPE = {10, 20, 30, 80}
MEMORY_ABNORMAL_TYPE = {50, 60}

errorFeatureDict = {
    10: ["process_cpu"],
    20: ["process_cpu"],
    30: ["process_cpu"],
    80: ["process_cpu"],
    50: ["server_pgfree"],
    60: ["server_used"],
}

# 定义一些数据的差分数据
usefulFeatures={
    "server": ["mem_used", "pgfree", "freq", "usr_cpu", "kernel_cpu"],
    "server_diff":  ["pgfree", "usr_cpu", "kernel_cpu"],
    "process": ["usr_cpu", "kernel_cpu", "rss", "read_chars", "read_bytes"],
    "process_diff": ["usr_cpu", "kernel_cpu", "read_chars", "read_bytes"],
    "compute": ["cpu_power", "power", "cabinet_power", "fan1_speed", "fan2_speed", "fan3_speed", "fan4_speed",
               "cpu1_core_rem", "cpu2_core_rem", "cpu3_core_rem", "cpu4_core_rem", "cpu1_mem_temp", "cpu2_mem_temp",
               "cpu3_mem_temp", "cpu4_mem_temp", "pch_temp"],
    "compute_diff": [],
    "network": ["tx_packets_phy", "rx_packets_phy"],
    "network_dff": ["tx_packets_phy", "rx_packets_phy"],
    "ping": ["avg_lat"],
    "ping_diff":[],
    "topdown": ["ddrc_rd", "ddrc_wr", "mflops"],
    "topdown_diff": []

}
