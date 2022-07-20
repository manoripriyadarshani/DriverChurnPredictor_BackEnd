import pandas as pd

from request.single_predict_request import SinglePredictRequest
from response.bulk_predict_response import PredictedModelDetail
from response.bulk_predict_response import BulkPredictResult
from response.single_predict_response import SinglePredictResponse
from util import single_predictor


class PredictorUseCase:

    def singlePredict(self, singlePredictRequest):
        isChurn, algo, accuracy, precision, recall, f1score = single_predictor.PredictSingleDriver(singlePredictRequest)
        return SinglePredictResponse(isChurn, algo, round(accuracy, 4), round(precision, 4), round(recall, 4),
                                     round(f1score, 4))

    def bulkPredict(self, bulkPredictRequest):
        df = pd.read_csv('/home/manorip/Documents/MSC/FYP/work_place/driverBulkData/driver_data.csv')
        results = []
        predictedModelDetail = None
        for index, row in df.iterrows():
            predictRequest = SinglePredictRequest(row['age'],
                                                  row['gender'],
                                                  row['latestLoyaltyLabel'],
                                                  row['calculatedLoyaltyLabel'],
                                                  row['vehicleType'],
                                                  row['serviceTime'],
                                                  row['driverRating'],
                                                  row['driverRaisedInquiryCount'],
                                                  row['passengerRaisedInquiryCount'],
                                                  row['severeInquiryCount'],
                                                  row['complainCount'],
                                                  row['monthlyAverageEarning'],
                                                  row['monthlyAverageRideDistance'],
                                                  row['monthlyAveragePasDiscount'],
                                                  row['monthlyAverageWaitingRides'],
                                                  row['lastMonthRejectedTripCount'],
                                                  row['IsTemporaryBlocked'],
                                                  row['IsUnAssigned'],
                                                  bulkPredictRequest.IsBestAlgo,
                                                  bulkPredictRequest.algo,
                                                  )
            isChurn, algo, accuracy, precision, recall, f1score = single_predictor.PredictSingleDriver(predictRequest)
            result = BulkPredictResult(isChurn, str(row['id']))
            if predictedModelDetail is None:
                predictedModelDetail = PredictedModelDetail(algo, round(accuracy, 4), round(precision, 4),
                                                            round(recall, 4), round(f1score, 4))
            results.append(result)

        return results, predictedModelDetail
