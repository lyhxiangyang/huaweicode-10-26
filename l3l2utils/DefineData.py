from enum import unique, IntEnum


@unique
class NodeState(IntEnum):
    pass



WINDOWS_SIZE= 5
FAULT_FLAG = "faultFlag"
TIME_COLUMN_NAME = "time"
CPU_FEATURE = "cpu_affinity"
PID_FEATURE = "pid"
TIME_INTERVAL = 60

PROCESS_CPUNAME = "cpu"
# 定义固定的文件名字
FDR = 0.01

# 存放模型的路径
SaveModelPath = 'Classifiers\\saved_model'

# 需要排除的特征名字
EXCLUDE_FEATURE_NAME = ["time", FAULT_FLAG]

# 机器学习用到的常数
# 模型类型
MODEL_TYPE = ['decision_tree', 'random_forest', 'adaptive_boosting']

# 属于CPU和MEMORY异常类型的种类
CPU_ABNORMAL_TYPE = [10, 20, 30, 80]
MEMORY_ABNORMAL_TYPE = [50, 60]
