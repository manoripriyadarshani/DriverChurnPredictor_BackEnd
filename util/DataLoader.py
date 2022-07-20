from datetime import datetime

import pandas as pd


def LoadAndAnalyzeData():

    path = "/home/manorip/Documents/MSC/FYP/work_place/dataset/"
    data_CRM_ticket = pd.read_csv(path + '/crm_tickets_last_6_months.csv')
    data_rides = pd.read_csv(path + '/rides_last_6_months.csv')
    data_driverIds = pd.read_csv(path + '/driverids.csv')
    data_driverDetails = pd.read_csv(path + '/driver_details.csv')
    data_driverRanking = pd.read_csv(path + '/driver_ranking.csv')

    ######################  CRM row data set --> CRM amalysed dataset   ###############################

    # get dataframe with only required columns
    df = data_CRM_ticket[
        ['ticket.id', 'driverid', 'ticket.createdatesk', 'ticket.title', 'ticket.category', 'ticket.status',
         'ticket.severity', 'ticket.raisedby', 'ticket.priority', 'ticket.iscomplain']]

    # number of tickets raised by the driver
    df1 = df[df["ticket.raisedby"] == 'Driver']
    df1_group = df1.groupby(["driverid"])["ticket.id"].count().reset_index(name="driver_raised_ticket_count")
    # print(df1_group)

    # get number of complains for a driver
    df2 = df[df["ticket.iscomplain"] == 1]
    df2_group = df2.groupby(["driverid"])["ticket.id"].count().reset_index(name="complain_ticket_count")
    # print(df2_group)

    # number of high seviour tickets
    df3 = df[df["ticket.severity"].isin(['High', 'Critical', 'Major'])]
    df3_group = df3.groupby(["driverid"])["ticket.id"].count().reset_index(name="severe_ticket_count")
    # print(df3_group )

    # number of tickets raised by the passenger
    df4 = df[df["ticket.raisedby"] == 'Passenger']
    df4_group = df4.groupby(["driverid"])["ticket.id"].count().reset_index(name="passenger_raised_ticket_count")
    # print(df4_group)

    # crm ticket merged
    CRM_data_merged = (data_driverIds.merge(df1_group, on='driverid', how='left')
                       .merge(df2_group, on='driverid', how='left')
                       .merge(df3_group, on='driverid', how='left')
                       .merge(df4_group, on='driverid', how='left'))

    #########################  Rides row data set --> Rides amalysed dataset   #####################################

    # print(data_rides.describe())
    data_rides["fact_rides.createdatesk"] = data_rides["fact_rides.createdatesk"].apply(
        lambda x: (datetime.strptime(str(x), "%Y%m%d").date()))
    data_rides['fact_rides.createdatesk'] = pd.to_datetime(data_rides['fact_rides.createdatesk'])

    group_monthly_earn = (data_rides[data_rides["fact_rides.tripfare"].notna()]
                          .groupby(['driverid', pd.Grouper(freq='m', key='fact_rides.createdatesk')])[
                              'fact_rides.tripfare'].sum()
                          .groupby(level=0)
                          .mean()
                          .reset_index(name='avg_monthly_earning'))

    group_monthly_ride_distance = (data_rides[data_rides["fact_rides.distance"].notna()]
                                   .groupby(['driverid', pd.Grouper(freq='m', key='fact_rides.createdatesk')])[
                                       'fact_rides.distance'].sum()
                                   .groupby(level=0)
                                   .mean()
                                   .reset_index(name='avg_monthly_ride_distance'))

    group_monthly_pass_discounts = (data_rides[data_rides["fact_rides.passengerdiscount"].notna()]
                                    .groupby(['driverid', pd.Grouper(freq='m', key='fact_rides.createdatesk')])[
                                        'fact_rides.passengerdiscount'].sum()
                                    .groupby(level=0)
                                    .mean()
                                    .reset_index(name='avg_monthly_passenger_discounts'))

    group_monthly_waiting_rides = (
        data_rides[data_rides["fact_rides.waitingcost"].notna() & data_rides["fact_rides.waitingcost"] > 0]
        .groupby(['driverid', pd.Grouper(freq='m', key='fact_rides.createdatesk')])[
            'fact_rides.passengerdiscount'].count()
        .groupby(level=0)
        .mean()
        .reset_index(name='avg_monthly_waiting_rides'))

    # crm ticket merged
    ride_data_merged = (data_driverIds.merge(group_monthly_earn, on='driverid', how='left')
                        .merge(group_monthly_ride_distance, on='driverid', how='left')
                        .merge(group_monthly_pass_discounts, on='driverid', how='left')
                        .merge(group_monthly_waiting_rides, on='driverid', how='left'))

    #############################################  merge data sets into one ###########################################

    driver_data_final = (data_driverIds.merge(data_driverDetails, left_on='driverid', right_on='driverid', how='left')
                         .merge(data_driverRanking, left_on='driverid', right_on='driverid', how='left')
                         .merge(CRM_data_merged, left_on='driverid', right_on='driverid', how='left')
                         .merge(ride_data_merged, left_on='driverid', right_on='driverid', how='left'))

    return driver_data_final
