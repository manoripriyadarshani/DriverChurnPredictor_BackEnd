from response.model_train_response import ModelTrainResult
from util import model_trainer


class ModelTrainerUseCase:

    def trainMLModel(self, modelTrainRequest, driverDataFinal):
        results = []
        for x in modelTrainRequest.algo:
            precision, accuracy, recall, f1 = model_trainer.TrainModelByAlgo(x, driverDataFinal)
            result = ModelTrainResult(round(precision, 4), round(accuracy, 4), round(recall, 4), round(f1, 4),x)
            results.append(result)

        return results
