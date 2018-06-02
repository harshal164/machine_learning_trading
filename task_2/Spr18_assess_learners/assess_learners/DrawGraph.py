"""
Test a learner.  (c) 2015 Tucker Balch
"""
import matplotlib.pyplot as plt
import numpy as np
import math
# import LinRegLearner as lrl
import DTLearner as dtl
from BagLearner import BagLearner
from DTLearner import DTLearner
import BagLearner as btl
# import InsaneLearner as it
# import RTLearner as rtl
import sys

if __name__=="__main__":
    print sys.argv
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    if sys.argv=='Istanbul.csv':
        inf=inf[1:-1]
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])
    # DTlearner
    # compute how much of the data is training and testing
    train_rows = int(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    print testX.shape
    print testY.shape
    leaf_size=[]
    rmse=[]
    rmse_test=[]
    # create a learner and train it
    for i in range(50):
        learner = BagLearner(learner=DTLearner ,kwargs={"leaf_size":i},bags=20,boost=False,verbose=False)  # constructor
        learner.addEvidence(trainX, trainY)  # training step
        Y = learner.query(trainX)  # query
        Y_test = learner.query(testX)  # query
        print learner.author()
        # learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
        # learner.addEvidence(trainX, trainY) # train it
        # pred=learner.query(trainX)
        # print learner.author()

        # evaluate in sample
        # predY = dt_learner.query(trainX) # get the predictions
        error = math.sqrt(((trainY - Y) ** 2).sum()/trainY.shape[0])
        error_test = math.sqrt(((testY - Y_test) ** 2).sum() / testY.shape[0])

        print
        print "In sample results"
        print "RMSE: ", rmse
        leaf_size.append(i)
        rmse.append(error)
        rmse_test.append(error_test)
        # c = np.corrcoef(pred, y=trainY)
        # print "corr: ", c[0,1]



    # evaluate out of sample
    # predY = dt_learner.query(testX) # get the predictions
    # rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    # print
    # print "Out of sample results"
    print "RMSE: ", rmse
    # c = np.corrcoef(predY, y=testY)
    # print "corr: ", c[0,1]
    print(leaf_size)

    plt.plot(leaf_size,rmse,'g--',label='Train')
    plt.plot(leaf_size, rmse_test,'r--',label='test')
    plt.savefig("final_100_overfit_1 ")
    plt.legend()
    plt.show()