class SinglePredictRequest():

    def __init__(self):
        pass

    def __init__(self, age, gender, latestLoyaltyLabel, calculatedLoyaltyLabel,vehicleType, serviceTime, driverRating,
                 driverRaisedInquiryCount,passengerRaisedInquiryCount,severeInquiryCount, complainCount, monthlyAverageEarning, monthlyAverageRideDistance,
                 monthlyAveragePasDiscount,monthlyAverageWaitingRides, lastMonthRejectedTripCount, IsTemporaryBlocked, IsUnAssigned, IsBestAlgo,algo):
        self.age = age
        self.gender = gender
        self.latestLoyaltyLabel = latestLoyaltyLabel
        self.calculatedLoyaltyLabel = calculatedLoyaltyLabel
        self.vehicleType = vehicleType
        self.serviceTime = serviceTime
        self.driverRating = driverRating
        self.driverRaisedInquiryCount = driverRaisedInquiryCount
        self.passengerRaisedInquiryCount = passengerRaisedInquiryCount
        self.severeInquiryCount = severeInquiryCount
        self.complainCount = complainCount
        self.monthlyAverageEarning = monthlyAverageEarning
        self.monthlyAverageRideDistance = monthlyAverageRideDistance
        self.monthlyAveragePasDiscount = monthlyAveragePasDiscount
        self.monthlyAverageWaitingRides = monthlyAverageWaitingRides
        self.lastMonthRejectedTripCount = lastMonthRejectedTripCount
        self.IsTemporaryBlocked = IsTemporaryBlocked
        self.IsUnAssigned = IsUnAssigned
        self.IsBestAlgo = IsBestAlgo
        self.algo = algo

    def __repr__(self):
        return '<PredictRequest(name={self.gender!r})>'.format(self=self)
