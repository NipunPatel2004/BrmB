import pickle

def load_models():

    model = pickle.load(open("Models/heart_disease_model.pkl", "rb"))
    scaler = pickle.load(open("Models/scaler.pkl", "rb"))
    accuracy = pickle.load(open("Models/accuracy.pkl", "rb"))
    report = pickle.load(open("Models/classification_report.pkl", "rb"))
    cm = pickle.load(open("Models/confusion_matrix.pkl", "rb"))
    return model, scaler, accuracy, report, cm