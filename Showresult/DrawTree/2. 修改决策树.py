from Classifiers.TrainToTest import show_threshold, change_threshold

if __name__ == '__main__':
    file = r'tmp/modelpath/l2/温度过高/decision_tree.pkl'
    show_threshold(file)
    change_threshold(file, 0, 500)
