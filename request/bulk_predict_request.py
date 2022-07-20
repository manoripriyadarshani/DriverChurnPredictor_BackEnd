class BulkPredictRequest():

    def __init__(self, IsBestAlgo, algo):
        self.IsBestAlgo = IsBestAlgo
        self.algo = algo

    def __repr__(self):
        return '<PredictRequest(name={self.gender!r})>'.format(self=self)
