from utils.DrawDecisionTree import plot_tree_structure

if __name__ == "__main__":
    # plot_tree_structure(modelpath="Classifiers/saved_model/tmp_load1_nosuffix")
    plot_tree_structure(modelpath="tmp/modelpath/singlefeature/memory_leak_model")
    plot_tree_structure(modelpath="tmp/modelpath/singlefeature/memory_bandwidth_model")
