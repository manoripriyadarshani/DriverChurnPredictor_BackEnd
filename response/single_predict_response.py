from flask import json

class SinglePredictResponse():
    def __init__(self, isChurn,  algorithm, accuracy, precision,   recall,  f1):
        self.isChurn = isChurn
        self.algorithm = algorithm
        self.modelPrecision = precision
        self.modelAccuracy = accuracy
        self.modelRecall = recall
        self.modelF1Score = f1

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


