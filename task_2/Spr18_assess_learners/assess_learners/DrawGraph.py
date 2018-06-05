"""
Test a learner.  (c) 2015 Tucker Balch
"""
import matplotlib.pyplot as plt
import numpy as np
import math
from time import time
import pandas

# from BagLearner import BagLearner
from DTLearner import DTLearner
import BagLearner as btl
from RTLearner import RTLearner

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
    rt_rmse=[]
    rmse_test=[]
    # create a learner and train it
    # for i in range(50):
    rt_learner=RTLearner(leaf_size=3)
    learner=DTLearner(leaf_size=3)
    start_time=time()
    learner.addEvidence(trainX, trainY)  # training step
    print("timeeeeeeeeeeeeeeeeee",time()-start_time)
    start_time=time()
    rt_learner.addEvidence(trainX, trainY)
    print("rt timeeeeeeeeeeeeeeeee",time()-start_time)
    Y = learner.query(trainX)  # query
    rt_Y=learner.query(trainX)  # query
    Y_test = learner.query(testX)  # query
    Y_test_rt = rt_learner.query(testX)  # query
    print("DT rmse",math.sqrt(((testY - Y_test) ** 2).sum() / testY.shape[0]))
    print("RT rmse", math.sqrt(((testY - Y_test_rt) ** 2).sum() / testY.shape[0]))
    rt_Y_test=learner.query(testX)  # query
    print learner.author()
    # learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
    # learner.addEvidence(trainX, trainY) # train it
    # pred=learner.query(trainX)
    # print learner.author()

    # evaluate in sample
    # predY = dt_learner.query(trainX) # get the predictions
    error = math.sqrt(((trainY - Y) ** 2).sum()/trainY.shape[0])
    rt_error = math.sqrt(((trainY - Y) ** 2).sum() / trainY.shape[0])
    # error_test = math.sqrt(((testY - Y_test) ** 2).sum() / testY.shape[0])

    # print
    # print "In sample results"
    # print "RMSE: ", rmse
    # leaf_size.append(i)
    # rmse.append(error)
    # rt_rmse.append(rt_error)
    # rmse_test.append(error_test)
    # c = np.corrcoef(pred, y=trainY)
    # print "corr: ", c[0,1]
    # line1=plt.plot(leaf_size,rmse,'r--')
    # line2=plt.plot(leaf_size, rt_rmse,'g--')
    # plt.legend()
    # plt.xlabel("leaf size")
    # plt.ylabel("error:(RMSE)")
    # plt.title('DTLearner vs RTLearner')
    #
    # plt.show()
    #
    # # df=pandas.DataFrame([leaf_size,rmse])
    # # plot=df.plot()
    # # plot.set_xlabel('Leaf size')
    # # plot.set_ylabel('Error')
    # # plot.set_title('Test for various leaf size in bagLearner')
    # # plt.show()
    # # evaluate out of sample
    # # predY = dt_learner.query(testX) # get the predictions
    # # rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    # # print
    # # print "Out of sample results"
    # # print "RMSE: ", rmse
    # # c = np.corrcoef(predY, y=testY)
    # # print "corr: ", c[0,1]
    # # print(leaf_size)
    #
    # # plt.plot(leaf_size,rmse,'g--',label='Train')
    # # plt.plot(leaf_size, rmse_test,'r--',label='test')
    # # plt.savefig("final_100_overfit_1 ")
    # # plt.legend()
    # # plt.show()