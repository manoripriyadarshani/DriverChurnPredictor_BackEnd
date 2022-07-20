import joblib
import pandas as pd

from util import model_details


def PredictSingleDriver(singlePredictRequest):
    if singlePredictRequest.IsBestAlgo:
        singlePredictRequest.algo = model_details.bestAlgo["algo"]

    switcher = {
        "NB": lambda: NB(singlePredictRequest),
        "KNN": lambda: KNN(singlePredictRequest),
        "LR": lambda: LR(singlePredictRequest),
        "SVM": lambda: SVM(singlePredictRequest),
        "DT": lambda: DT(singlePredictRequest),
        "RF": lambda: RF(singlePredictRequest),
        "FFNN": lambda: FFNN(singlePredictRequest),
        "ST_NB_KNN_WITH_LR": lambda: ST_NB_KNN_WITH_LR(singlePredictRequest),
        "ST_DT_SVM_RF_LR": lambda: ST_DT_SVM_RF_LR(singlePredictRequest),
        "BG_DT": lambda: BG_DT(singlePredictRequest),
        "XGB_DT": lambda: XGB_DT(singlePredictRequest),
        "ADAB_DT": lambda: ADAB_DT(singlePredictRequest),
        "ADAB_RF": lambda: ADAB_RF(singlePredictRequest)
    }
    func = switcher.get(singlePredictRequest.algo, lambda: "nothing")
    return func()


def LR(request):
    reloaded_model = joblib.load('/home/manorip/Documents/MSC/FYP/work_place/model_store/LR_Model.pkl')
    new_input = getNewInput(request)
    new_output = reloaded_model.predict(new_input)
    model_detail = model_details.algoAccuracies["LR"]
    is_churn = True if new_output[0] == 1 else False
    return is_churn, "LR", model_detail.accuracy, model_detail.precision, model_detail.recall, model_detail.f1score


def KNN(request):
    reloaded_model = joblib.load('/home/manorip/Documents/MSC/FYP/work_place/model_store/KNN_Model.pkl')
    new_input = getNewInput(request)
    new_output = reloaded_model.predict(new_input)
    model_detail = model_details.algoAccuracies["KNN"]
    is_churn = True if new_output[0] == 1 else False
    return is_churn, "KNN", model_detail.accuracy, model_detail.precision, model_detail.recall, model_detail.f1score


def NB(request):
    reloaded_model = joblib.load('/home/manorip/Documents/MSC/FYP/work_place/model_store/NB_Model.pkl')
    new_input = getNewInput(request)
    new_output = reloaded_model.predict(new_input)
    model_detail = model_details.algoAccuracies["NB"]
    is_churn = True if new_output[0] == 1 else False
    return is_churn, "NB", model_detail.accuracy, model_detail.precision, model_detail.recall, model_detail.f1score

def SVM(request):
    reloaded_model = joblib.load('/home/manorip/Documents/MSC/FYP/work_place/model_store/SVM_Model.pkl')
    new_input = getNewInput(request)
    new_output = reloaded_model.predict(new_input)
    model_detail = model_details.algoAccuracies["SVM"]
    is_churn = True if new_output[0] == 1 else False
    return is_churn, "SVM", model_detail.accuracy, model_detail.precision, model_detail.recall, model_detail.f1score


def DT(request):
    reloaded_model = joblib.load('/home/manorip/Documents/MSC/FYP/work_place/model_store/DT_Model.pkl')
    new_input = getNewInput(request)
    new_output = reloaded_model.predict(new_input)
    model_detail = model_details.algoAccuracies["DT"]
    is_churn = True if new_output[0] == 1 else False
    return is_churn, "DT", model_detail.accuracy, model_detail.precision, model_detail.recall, model_detail.f1score


def RF(request):
    reloaded_model = joblib.load('/home/manorip/Documents/MSC/FYP/work_place/model_store/RF_Model.pkl')
    new_input = getNewInput(request)
    new_output = reloaded_model.predict(new_input)
    model_detail = model_details.algoAccuracies["RF"]
    is_churn = True if new_output[0] == 1 else False
    return is_churn, "RF", model_detail.accuracy, model_detail.precision, model_detail.recall, model_detail.f1score


# todo
def FFNN(request):
    reloaded_model = joblib.load('/home/manorip/Documents/MSC/FYP/work_place/model_store/NB_Model.pkl')
    new_input = getNewInput(request)
    new_output = reloaded_model.predict(new_input)
    model_detail = model_details.algoAccuracies["FFNN"]
    is_churn = True if new_output[0] == 1 else False
    return is_churn, "FFNN", model_detail.accuracy, model_detail.precision, model_detail.recall, model_detail.f1score


def ST_NB_KNN_WITH_LR(request):
    reloaded_model = joblib.load('/home/manorip/Documents/MSC/FYP/work_place/model_store/ST_NB_KNN_WITH_LR_Model.pkl')
    new_input = getNewInput(request)
    new_output = reloaded_model.predict(new_input)
    model_detail = model_details.algoAccuracies["ST_NB_KNN_WITH_LR"]
    is_churn = True if new_output[0] == 1 else False
    return is_churn, "ST_NB_KNN_WITH_LR", model_detail.accuracy, model_detail.precision, model_detail.recall, model_detail.f1score


def ST_DT_SVM_RF_LR(request):
    reloaded_model = joblib.load('/home/manorip/Documents/MSC/FYP/work_place/model_store/ST_DT_SVM_RF_LR_Model.pkl')
    new_input = getNewInput(request)
    new_output = reloaded_model.predict(new_input)
    model_detail = model_details.algoAccuracies["ST_DT_SVM_RF_LR"]
    is_churn = True if new_output[0] == 1 else False
    return is_churn, "ST_DT_SVM_RF_LR", model_detail.accuracy, model_detail.precision, model_detail.recall, model_detail.f1score


# todo
def BG_DT(request):
    reloaded_model = joblib.load('/home/manorip/Documents/MSC/FYP/work_place/model_store/BG_DT_Model.pkl')
    new_input = getNewInput(request)
    new_output = reloaded_model.predict(new_input)
    model_detail = model_details.algoAccuracies["BG_DT"]
    is_churn = True if new_output[0] == 1 else False
    return is_churn, "BG_DT", model_detail.accuracy, model_detail.precision, model_detail.recall, model_detail.f1score


def XGB_DT(request):
    reloaded_model = joblib.load('/home/manorip/Documents/MSC/FYP/work_place/model_store/XGB_DT_Model.pkl')
    new_input = getNewInput(request)
    new_output = reloaded_model.predict(new_input)
    model_detail = model_details.algoAccuracies["XGB_DT"]
    is_churn = True if new_output[0] == 1 else False
    return is_churn, "XGB_DT", model_detail.accuracy, model_detail.precision, model_detail.recall, model_detail.f1score


def ADAB_DT(request):
    reloaded_model = joblib.load('/home/manorip/Documents/MSC/FYP/work_place/model_store/ADAB_DT_Model.pkl')
    new_input = getNewInput(request)
    new_output = reloaded_model.predict(new_input)
    model_detail = model_details.algoAccuracies["ADAB_DT"]
    is_churn = True if new_output[0] == 1 else False
    return is_churn, "ADAB_DT", model_detail.accuracy, model_detail.precision, model_detail.recall, model_detail.f1score


def ADAB_RF(request):
    reloaded_model = joblib.load('/home/manorip/Documents/MSC/FYP/work_place/model_store/ADAB_RF_Model.pkl')
    new_input = getNewInput(request)
    new_output = reloaded_model.predict(new_input)
    model_detail = model_details.algoAccuracies["ADAB_RF"]
    is_churn = True if new_output[0] == 1 else False
    return is_churn, "ADAB_RF", model_detail.accuracy, model_detail.precision, model_detail.recall, model_detail.f1score


def getNewInput(request):
    calculatedLoyaltyLabel_is_G = 1 if request.calculatedLoyaltyLabel == 'gold' else 0
    calculatedLoyaltyLabel_is_P = 1 if request.calculatedLoyaltyLabel == 'platinum' else 0
    calculatedLoyaltyLabel_is_R = 1 if request.calculatedLoyaltyLabel == 'regular' else 0
    calculatedLoyaltyLabel_is_S = 1 if request.calculatedLoyaltyLabel == 'silver' else 0

    latestLoyaltyLabel_is_D = 1 if request.latestLoyaltyLabel == 'dedicated' else 0
    latestLoyaltyLabel_is_G = 1 if request.latestLoyaltyLabel == 'gold' else 0
    latestLoyaltyLabel_is_P = 1 if request.latestLoyaltyLabel == 'platinum' else 0
    latestLoyaltyLabel_is_R = 1 if request.latestLoyaltyLabel == 'regular' else 0
    latestLoyaltyLabel_is_S = 1 if request.latestLoyaltyLabel == 'silver' else 0

    gender_is_1 = 1 if request.gender == 'male' else 0
    gender_is_2 = 1 if request.gender == 'female' else 0

    isTempBlocked = 1 if request.IsTemporaryBlocked else 0
    isUnAssigned = 1 if request.IsUnAssigned else 0

    new_input = [[request.serviceTime,
                  request.driverRating,
                  request.lastMonthRejectedTripCount,
                  request.vehicleType,
                  request.driverRaisedInquiryCount,
                  request.complainCount,
                  request.severeInquiryCount,
                  request.passengerRaisedInquiryCount,
                  request.monthlyAverageEarning,
                  request.monthlyAverageRideDistance,
                  request.monthlyAveragePasDiscount,
                  request.monthlyAverageWaitingRides,
                  request.age,
                  gender_is_1,
                  gender_is_2,
                  calculatedLoyaltyLabel_is_G,
                  calculatedLoyaltyLabel_is_P,
                  calculatedLoyaltyLabel_is_R,
                  calculatedLoyaltyLabel_is_S,
                  latestLoyaltyLabel_is_D,
                  latestLoyaltyLabel_is_G,
                  latestLoyaltyLabel_is_P,
                  latestLoyaltyLabel_is_R,
                  latestLoyaltyLabel_is_S,
                  isTempBlocked,
                  isUnAssigned
                  ]]

    new_input_columns = ['service_time_days',
                         'latest_driver_rating',
                         'rejected_trip_count_of_last_30_days',
                         'primary_taxi_model',
                         'driver_raised_ticket_count',
                         'complain_ticket_count',
                         'severe_ticket_count',
                         'passenger_raised_ticket_count',
                         'avg_monthly_earning',
                         'avg_monthly_ride_distance',
                         'avg_monthly_passenger_discounts',
                         'avg_monthly_waiting_rides',
                         'age',
                         'gender_1.0',
                         'gender_2.0',
                         'calculated_loyalty_label_Gold',
                         'calculated_loyalty_label_Platinum',
                         'calculated_loyalty_label_Regular',
                         'calculated_loyalty_label_Silver',
                         'latest_loyalty_label_Dedicated',
                         'latest_loyalty_label_Gold',
                         'latest_loyalty_label_Platinum',
                         'latest_loyalty_label_Regular',
                         'latest_loyalty_label_Silver',
                         'status_D',
                         'status_U'
                         ]

    return pd.DataFrame(new_input, columns=new_input_columns)
