from Classifiers.TrainToTest import show_threshold, change_threshold

if __name__ == '__main__':
    # 温度过高由于模型训练的是相当于低强度，我们这里是高强度  所以修改原始数据集
    file = r'tmp/modelpath/l2/温度过高/decision_tree.pkl'
    show_threshold(file)
    change_threshold(file, 0, 150.1)

    # 修改pfs网络异常，原始是121  改为200
    file = r'tmp/modelpath/l2/网络异常1/decision_tree.pkl'
    show_threshold(file)
    change_threshold(file, 0, 200)

    # 机器功率封顶原本设置为98.5 根据指标是cpu_pwer 但是现在要设置为70
    file = r'tmp/modelpath/l2/机器功率封顶/decision_tree.pkl'
    show_threshold(file)
    change_threshold(file, 0, 70)
