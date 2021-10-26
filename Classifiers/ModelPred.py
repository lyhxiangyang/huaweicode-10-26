import joblib

from utils.DefineData import SaveModelPath


def model_pred(x_pred, model_type, saved_model_path = SaveModelPath):
    """
    Use trained model to predict
    :param saved_model_path:
    :param x_pred: Samples to be predicted
    :param model_type: The type of saved model to be used
    :return y_pred: Class labels for samples in x_pred
    """
    # Load saved model
    model = joblib.load('%s\\%s.pkl' % (saved_model_path, model_type))

    y_pred = model.predict(x_pred)
    return y_pred


def select_and_pred(df, model_type, saved_model_path = SaveModelPath):
    # Select needed features
    f = open('%s\\header.txt' % saved_model_path, 'r')
    features = f.read().splitlines()
    df_selected = df[features]

    # Use trained model to predict
    y_pred = model_pred(df_selected, model_type, saved_model_path=saved_model_path)
    return y_pred
