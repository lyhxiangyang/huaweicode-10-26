from collections import defaultdict
from typing import List

"""
# --函数功能：
传入实际值列表和预测值列表，输出准确率、召回率等信息

返回值：
TP: 将正样本预测为正样本的数量
FN：将正样本预测为负样本的数量
FP：将负样本预测为正样本的数量
TN：将负样本预测为副样本的数量
precision: TP / (TP + FP)   正确预测正样本 占 全部预测正样本的比值   精确率
recall: TP / (TP + FN) 正确预测正样本 占 正样本总数的比值           召回率
accuracy: 准确率(accuracy) = 预测对的/所有 = (TP+TN)/(TP+FN+FP+TN) = 70%

"""


def get_metrics(reallist: List, prelist: List, label: int):
    """
    Get Statistical metrics
    :param reallist: Label list of samples
    :param prelist: Predicted label list
    :param label: Label to be selected
    :return: tp, fp, fn and tn etc
    """
    if len(reallist) != len(prelist):
        print("预测列表和真实标签的列表长度不一致")
        exit(1)
    true_pos, true_neg = 0, 0
    false_pos, false_neg = 0, 0
    rightnumber = 0
    for i in range(len(reallist)):
        if reallist[i] == prelist[i]:
            rightnumber += 1
        if prelist[i] == label:
            if reallist[i] == prelist[i]:
                # 正-正
                true_pos += 1
            else:
                # 负-正
                false_pos += 1
        else:
            if reallist[i] != label:
                # 负-负
                true_neg += 1
            else:
                # 正-负
                false_neg += 1
    precision = float('nan') if true_pos + false_pos == 0 else true_pos / (true_pos + false_pos)
    recall = float('nan') if true_pos + false_neg == 0 else true_pos / (true_pos + false_neg)
    accuracy = rightnumber / len(reallist)
    metrics = dict(tp=true_pos, tn=true_neg, fp=false_pos, fn=false_neg, precision=precision, recall=recall,
                   accuracy=accuracy)
    # 添加数量数据，即各个实际标签对应的数量
    realnums = {} # 总数量
    # 假设当前预测的的标签是11
    num_pre_itself = 0 # 表示预测值为11的数量
    num_pre_normal = 0 # 表示预测值为0的数量
    num_pre_samefault = 0 # 表示预测值为11、12、13、14、15的数量
    num_pre_fault = 0 # 表示预测为非0的数量

    NORMAL_LABEL = 0 # 代表正常类型的标签
    for i in range(len(reallist)):
        if reallist[i] not in realnums.keys():
            realnums[reallist[i]] = 0
        realnums[reallist[i]] += 1
        # 正常的数量
        # if prelist[i] not in numDict[i].keys():
        #     numDict[i]
        if reallist[i] != label:
            continue
        if prelist[i] == label:
            num_pre_itself += 1
        if prelist[i] == NORMAL_LABEL:
            num_pre_normal += 1
        if prelist[i] // 10 == label // 10:
            num_pre_samefault += 1
        if prelist[i] != NORMAL_LABEL:
            num_pre_fault += 1


    metrics["realnums"] = realnums
    # 假设我们预测的标签时11
    labelnum = realnums[label]
    metrics["per_itself"] = num_pre_itself / labelnum # 预测为11的百分比
    metrics["per_normal"] = num_pre_normal / labelnum # 预测为0的百分比
    metrics["per_samefault"] = num_pre_samefault / labelnum # 预测为11、12、13、14、15的百分比
    metrics["per_fault"] = num_pre_fault / labelnum # 预测为非0的百分比
    # print("统计数据-{}".format(label).center(40, "*"))
    # print("num: {}".format(labelnum))
    # print("num_normal: {}".format(num_pre_normal))
    # print("num_fault: {}".format(num_pre_fault))
    # print("统计数据结束".center(40, "*"))

    return metrics
