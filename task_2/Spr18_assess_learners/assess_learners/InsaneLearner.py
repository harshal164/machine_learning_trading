import BagLearner as bl,LinRegLearner as lrl,numpy as np
class InsaneLearner(object):
    def __init__(self, verbose):
        learner_list = []
        no_of_bags = 20
        for i in range(no_of_bags):
            learner_list.append(bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20,verbose=verbose))
        self.learner_list = learner_list
        self.no_of_bags = no_of_bags
    def addEvidence(self, trainX, trainY):
        for learner in self.learner_list:
            learner.addEvidence(trainX, trainY)
    def query(self, testX):
        pred = np.empty((testX.shape[0], self.no_of_bags))
        for col in range(self.no_of_bags):
            pred[:, col] = self.learner_list[col].query(testX)
        return pred.mean(axis=1)
    def author(self):
        return 'harshal'