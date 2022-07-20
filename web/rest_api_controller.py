from flask import Flask, json
from flask_restful import Api, request
from flask_cors import CORS

from request.bulk_predict_request import BulkPredictRequest
from request.model_train_request import ModelTrainRequest
from request.single_predict_request import SinglePredictRequest
from response.bulk_predict_response import BulkPredictResponse
from response.model_train_response import ModelTrainResponse
from usecases.model_trainer_usecase import ModelTrainerUseCase
from usecases.predictor_usecase import PredictorUseCase
from util import DataLoader, DataPreProcessor

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)


@app.route('/model/train', methods=['POST'])
def train():
    content = request.get_json()
    algo = content['algo']
    modelTrainRequest = ModelTrainRequest(algo)

    driverDataFinal = DataLoader.LoadAndAnalyzeData()
    dataPreProcessed = DataPreProcessor.PreProcessData(driverDataFinal)

    results = ModelTrainerUseCase().trainMLModel(modelTrainRequest, dataPreProcessed)
    return ModelTrainResponse(results).toJSON()


@app.route('/driver/predict', methods=['POST'])
def predictDriver():
    content = request.get_json()

    predictRequest = SinglePredictRequest(content['age'],
                                          content['gender'],
                                          content['latestLoyaltyLabel'],
                                          content['calculatedLoyaltyLabel'],
                                          content['vehicleType'],
                                          content['serviceTime'],
                                          content['driverRating'],
                                          content['driverRaisedInquiryCount'],
                                          content['passengerRaisedInquiryCount'],
                                          content['severeInquiryCount'],
                                          content['complainCount'],
                                          content['monthlyAverageEarning'],
                                          content['monthlyAverageRideDistance'],
                                          content['monthlyAveragePasDiscount'],
                                          content['monthlyAverageWaitingRides'],
                                          content['lastMonthRejectedTripCount'],
                                          content['IsTemporaryBlocked'],
                                          content['IsUnAssigned'],
                                          content['IsBestAlgo'],
                                          content['algo'],
                                          )

    return PredictorUseCase().singlePredict(predictRequest).toJSON()


@app.route('/drivers/predict', methods=['POST'])
def predictDrivers():
    content = request.get_json()

    predictRequest = BulkPredictRequest(content['IsBestAlgo'], content['algo'], )
    results, predictedModelDetails = PredictorUseCase().bulkPredict(predictRequest)
    return BulkPredictResponse(results, predictedModelDetails).toJSON()


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)
