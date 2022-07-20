class ModelDetail():
    def __init__(self, accuracy, precision, recall, f1score):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1score = f1score


bestAlgo = {
    "algo": "RF",
    "accuracy": 0.0,
}

algoAccuracies = {
    "NB": ModelDetail(0.0, 0.0, 0.0, 0.0),
    "KNN": ModelDetail(0.0, 0.0, 0.0, 0.0),
    "LR": ModelDetail(0.0, 0.0, 0.0, 0.0),
    "SVM": ModelDetail(0.0, 0.0, 0.0, 0.0),
    "DT": ModelDetail(0.0, 0.0, 0.0, 0.0),
    "RF": ModelDetail(0.0, 0.0, 0.0, 0.0),
    "FFNN": ModelDetail(0.0, 0.0, 0.0, 0.0),
    "ST_NB_KNN_WITH_LR": ModelDetail(0.0, 0.0, 0.0, 0.0),
    "ST_DT_SVM_RF_LR": ModelDetail(0.0, 0.0, 0.0, 0.0),
    "BG_DT": ModelDetail(0.0, 0.0, 0.0, 0.0),
    "XGB_DT": ModelDetail(0.0, 0.0, 0.0, 0.0),
    "ADAB_DT": ModelDetail(0.0, 0.0, 0.0, 0.0),
    "ADAB_RF": ModelDetail(0.0, 0.0, 0.0, 0.0),
}


def SetBestAlgo(algo, accuracy):
    bestAlgo["algo"] = algo
    bestAlgo["accuracy"] = accuracy

