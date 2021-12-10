from Classifiers.TrainToTest import show_threshold, change_threshold

if __name__ == '__main__':
    file = r'tmp/modelpath/singlefeature/memory_bandwidth_model/decision_tree.pkl'
    show_threshold(file)
    change_threshold(file, 0, 500)
