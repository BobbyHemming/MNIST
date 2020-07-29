import joblib


def save_model(model, name):
    joblib.dump(model, f'{name}.pkl')


def load_model(name):
    return joblib.load(f'{name}.pkl')