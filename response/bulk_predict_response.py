from flask import json

class BulkPredictResult():
    def __init__(self, isChurn, id):
        self.isChurn = isChurn
        self.id = id

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class PredictedModelDetail():
    def __init__(self, algo, accuracy, precision, recall,f1score):
        self.algorithm = algo
        self.modelAccuracy = accuracy
        self.modelPrecision = precision
        self.modelRecall = recall
        self.modelF1Score = f1score

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class BulkPredictResponse():
    def __init__(self, bulkPredictResults, predictedModelDetails):
        self.bulkPredictResults = bulkPredictResults
        self.predictedModelDetails = predictedModelDetails

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)



