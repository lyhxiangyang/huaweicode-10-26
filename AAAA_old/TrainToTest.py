# 使用三种模型进行预测信息

import joblib

def show_threshold(filepath):
    f = open(filepath, 'rb')
    model = joblib.load(f)
    f.close()
    print('node features  :', model.tree_.feature)
    print('node thresholds:', model.tree_.threshold)


def change_threshold(filepath, node, value):
    f = open(filepath, 'rb')
    model = joblib.load(f)
    f.close()
    model.tree_.threshold[node] = value
    joblib.dump(model, filepath)
    print('features and thresholds after modification:')
    show_threshold(filepath)

# if __name__ == '__main__':
#     file = r'hjx\decision_tree.pkl'
#     show_threshold(file)
#     change_threshold(file, 0, 106)
