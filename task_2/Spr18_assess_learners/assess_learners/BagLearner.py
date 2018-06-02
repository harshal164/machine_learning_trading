import numpy as np

class BagLearner(object):

    def __init__(self, learner, kwargs={}, bags=1, boost=False, verbose=False):
        self.verbose = verbose
        if self.verbose == True:
            for name, value in kwargs.items():  # print name-value pairs contained in kwargs dictionary
                print '{0} = {1}'.format(name, value)

        learnerList = []
        for bag in range(bags):
            learnerList.append(learner(verbose=self.verbose, **kwargs))

        self.learnerList = learnerList
        self.bags = bags
        self.boost = boost

    def addEvidence(self, trainX, trainY):
        bagSize = len(trainY)
        for learner in self.learnerList:
            ix = np.random.choice(range(bagSize), bagSize, replace=True)
            bagX = trainX[ix];
            bagY = trainY[ix]
            learner.addEvidence(bagX, bagY)  # add training examples/labels to each BagLearner

    def author(self):
        return 'Harshal'  # replace tb34 with your Georgia Tech username


    def query(self, testX):
        pred = np.empty((testX.shape[0], self.bags))
        for col in range(pred.shape[1]):
            pred[:, col] = self.learnerList[col].query(testX)
        return np.mean(pred, axis=1)


