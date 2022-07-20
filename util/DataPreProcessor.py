from datetime import date
from datetime import datetime

import numpy as np
import pandas as pd


def PreProcessData(data):

    ############################ Data transformation  ###########################
    # transform DOB to age
    data["age"] = data["dateofbirth"].apply(lambda x: (age(x)))
    data = data.drop(columns=['dateofbirth'])

    # round taxi model to int
    data["primary_taxi_model"] = data["primary_taxi_model"].apply(np.floor)

    ############################ Data Imputation  ###############################

    cols_mean_imputation = ["service_time_days", "latest_driver_rating", "passenger_raised_ticket_count",
                            "driver_raised_ticket_count",
                            "complain_ticket_count", "severe_ticket_count", "rejected_trip_count_of_last_30_days",
                            "completed_trip_count_of_last_30_days",
                            "completed_trip_count_of_last_180_days", "completed_trip_count_of_last_360_days",
                            "completion_ratio_of_last_30_days",
                            "avg_monthly_earning", "avg_monthly_ride_distance", "avg_monthly_waiting_rides",
                            "avg_monthly_passenger_discounts"]

    cols_mode_imputation = ["gender", "primary_taxi_model", "previous_loyalty_label", "calculated_loyalty_label",
                            "latest_loyalty_label"]

    data[cols_mode_imputation] = data[cols_mode_imputation].fillna(data.mode().iloc[0])
    data[cols_mean_imputation] = data[cols_mean_imputation].fillna(data[cols_mean_imputation].mean())
    data["age"] = data["age"].replace(0, data["age"].mean())

    lowerbound, upperbound = outlier_treatment(data.age, 1.5)
    data.drop(data[(data.age < lowerbound)].index, inplace=True)

    lowerbound, upperbound = outlier_treatment(data.latest_driver_rating, 1.5)
    data.drop(data[(data.latest_driver_rating > upperbound)].index, inplace=True)

    lowerbound, upperbound = outlier_treatment(data.avg_monthly_earning, 16.5)
    data.drop(data[(data.avg_monthly_earning > upperbound)].index, inplace=True)

    lowerbound, upperbound = outlier_treatment(data.avg_monthly_passenger_discounts, 25)
    data.drop(data[(data.avg_monthly_passenger_discounts > upperbound)].index, inplace=True)

    ####################### Categorical to numerical transformation  ################
    data = data.drop(columns=[ 'completed_trip_count_of_last_360_days',
                              'completed_trip_count_of_last_180_days', 'completion_ratio_of_last_30_days',
                              'previous_loyalty_label'])

    # onehot encoding for categorical columns
    data = pd.get_dummies(data, columns=['gender'])
    data = pd.get_dummies(data, columns=['calculated_loyalty_label'])
    data = pd.get_dummies(data, columns=['latest_loyalty_label'])
    data = pd.get_dummies(data, columns=['status'])
    data["churn"] = data["completed_trip_count_of_last_30_days"].apply(lambda x: (getChurn(x)))
    data = data.drop( columns=['status_PD', 'status_A','completed_trip_count_of_last_30_days'])

    ############################ Feature Selection  #################################

    # remove constant columns
    data = data.loc[:, data.apply(pd.Series.nunique) != 1]
    # print(driver_data_final.columns ) two columns removed
    # remove unique columns
    data = data.drop(columns=['driverid'])

    # get Correlation Metrix
    # plot.figure(figsize=(17,14))
    # cor = driver_data_final.corr()
    # seaborn.heatmap(cor, annot=True, cmap=plot.cm.CMRmap_r)
    # plot.show()

    # remove correlations
    corr_features = correlation(data, 0.8)
    driver_data_final = data.drop(corr_features, axis=1)

    return data


def age(born):
    if pd.isna(born):
        return 0
    else:
        born = datetime.strptime(born, "%Y-%m-%d %H:%M:%S").date()
        today = date.today()
        return today.year - born.year - ((today.month, today.day) < (born.month,born.day))

def getChurn(count):
     if count == 0:
        return 0
     else :
        return 1

def outlier_treatment(datacolumn,constant):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn , [25,75])
    IQR = Q3-Q1
    lower_range = Q1 - (constant * IQR)
    upper_range = Q3 + (constant * IQR)
    return lower_range,upper_range

# with the following function we can select highly correlated features
# it will remove the first feature that is correlated with anything other feature
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr
