from flask import json


class ModelTrainResponse():
    def __init__(self, modelTrainResults):
        self.modelTrainResults = modelTrainResults

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class ModelTrainResult():
    class_counter = 0
    def __init__(self, precision, accuracy, recall, f1, algorithm):
        self.modelPrecision = precision
        self.modelAccuracy = accuracy
        self.modelRecall = recall
        self.modelF1Score = f1
        self.algorithm = algorithm
        self.id = ModelTrainResult.class_counter
        ModelTrainResult.class_counter += 1

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
