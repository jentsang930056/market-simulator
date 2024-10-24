

import math
import sys
import numpy as np
import DTLearner as dtl
import RTLearner as rtl
import BagLearner as bl
import LinRegLearner as lrl
import matplotlib.pyplot as plt
import time


# Experiment 1
def exp1(train_x, train_y, test_x, test_y):
    lsize = list(range(1, 101))
    rmse_in_values = []
    rmse_out_values = []

    for leaf_size in lsize:
        learner = dtl.DTLearner(leaf_size=leaf_size, verbose=False)
        learner.add_evidence(train_x, train_y)

        pred_y_in = learner.query(train_x)
        pred_y_out = learner.query(test_x)
        rmse_in = math.sqrt(((train_y - pred_y_in) ** 2).sum() / train_y.shape[0])
        rmse_out = math.sqrt(((test_y - pred_y_out) ** 2).sum() / test_y.shape[0])
        rmse_in_values.append(rmse_in)
        rmse_out_values.append(rmse_out)

    plt.plot(lsize, rmse_in_values)
    plt.plot(lsize, rmse_out_values)
    plt.title("Experiment 1: RMSE vs leaf_size (No Bagging)")
    plt.xlabel("leaf size")
    plt.ylabel("RMSE")
    plt.legend(['In Sample RMSE', 'Out Sample RMSE'])
    plt.savefig("images/Figure 1.png")
    plt.close()


def exp2(train_x, train_y, test_x, test_y, bag_size=20):

    lsize = list(range(1, 101))
    rmse_in_values = []
    rmse_out_values = []

    for leaf_size in lsize:
        learner = bl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size": leaf_size}, bags=bag_size, verbose=False)
        learner.add_evidence(train_x, train_y)

        pred_y_in = learner.query(train_x)
        pred_y_out = learner.query(test_x)
        rmse_in = math.sqrt(((train_y - pred_y_in) ** 2).sum() / train_y.shape[0])
        rmse_out = math.sqrt(((test_y - pred_y_out) ** 2).sum() / test_y.shape[0])
        rmse_in_values.append(rmse_in)
        rmse_out_values.append(rmse_out)

    plt.plot(lsize, rmse_in_values)
    plt.plot(lsize, rmse_out_values)
    plt.title("Experiment 2: RMSE vs leaf_size (with Bagging)")
    plt.xlabel("leaf size")
    plt.ylabel("RMSE")
    plt.legend(['In Sample RMSE', 'Out Sample RMSE'])
    plt.savefig("images/Figure 2.png")
    plt.close()


def exp2(train_x, train_y, test_x, test_y):
    lsize = list(range(1, 101))
    rmse_in_values = []
    rmse_out_values = []

    for leaf_size in lsize:
        learner = bl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size": leaf_size}, bags=5, verbose=False)
        learner.add_evidence(train_x, train_y)

        pred_y_in = learner.query(train_x)
        pred_y_out = learner.query(test_x)
        rmse_in = math.sqrt(((train_y - pred_y_in) ** 2).sum() / train_y.shape[0])
        rmse_out = math.sqrt(((test_y - pred_y_out) ** 2).sum() / test_y.shape[0])
        rmse_in_values.append(rmse_in)
        rmse_out_values.append(rmse_out)

    plt.plot(lsize, rmse_in_values)
    plt.plot(lsize, rmse_out_values)
    plt.title("Experiment 2: RMSE vs leaf_size (with Bagging 5 bags)")
    plt.xlabel("leaf_size")
    plt.ylabel("RMSE")
    plt.legend(['In Sample RMSE', 'Out Sample RMSE'])
    plt.savefig("images/Figure 2.png")
    plt.close()


def exp2_2(train_x, train_y, test_x, test_y):
    lsize = list(range(1, 101))
    rmse_in_values = []
    rmse_out_values = []

    for leaf_size in lsize:
        learner = bl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size": leaf_size}, bags=20, verbose=False)
        learner.add_evidence(train_x, train_y)

        pred_y_in = learner.query(train_x)
        pred_y_out = learner.query(test_x)
        rmse_in = math.sqrt(((train_y - pred_y_in) ** 2).sum() / train_y.shape[0])
        rmse_out = math.sqrt(((test_y - pred_y_out) ** 2).sum() / test_y.shape[0])
        rmse_in_values.append(rmse_in)
        rmse_out_values.append(rmse_out)

    plt.plot(lsize, rmse_in_values)
    plt.plot(lsize, rmse_out_values)
    plt.title("Experiment 2: RMSE vs leaf_size (with Bagging 20 bags)")
    plt.xlabel("leaf_size")
    plt.ylabel("RMSE")
    plt.legend(['In Sample RMSE', 'Out Sample RMSE'])
    plt.savefig("images/Figure 3.png")
    plt.close()


def exp3(train_x, train_y, test_x, test_y):
    lsize = list(range(1, 51))
    training_time_dtl = []
    training_time_rtl = []
    mae_dtl = []
    mae_rtl = []

    for leaf_size in lsize:  # leaf_size from 1 to 50

        # training time and MAE for DTLearner
        start_time = time.time()
        learner_dtl = dtl.DTLearner(leaf_size=leaf_size, verbose=False)
        learner_dtl.add_evidence(train_x, train_y)
        end_time = time.time()
        training_time_dtl.append(end_time - start_time) #get training time

        pred_y_dtl = learner_dtl.query(test_x)
        mae_dtl.append(np.mean(np.abs(test_y - pred_y_dtl))) #get mae

        # training time and MAE for RTLearner
        start_time = time.time()
        learner_rtl = rtl.RTLearner(leaf_size=leaf_size, verbose=False)
        learner_rtl.add_evidence(train_x, train_y)
        end_time = time.time()
        training_time_rtl.append(end_time - start_time) #get training time

        pred_y_rtl = learner_rtl.query(test_x)
        mae_rtl.append(np.mean(np.abs(test_y - pred_y_rtl))) #get mae

    # Training Time Comparison
    plt.plot(lsize, training_time_dtl)
    plt.plot(lsize, training_time_rtl)
    plt.title("Experiment 3 : DT/RT Learner-Training Time")
    plt.xlabel("leaf size")
    plt.ylabel("training time")
    plt.legend(['DTLearner', 'RTLearner'])
    plt.savefig("images/Figure 4.png")
    plt.close()

    # MAE Comparison
    plt.plot(lsize, mae_dtl)
    plt.plot(lsize, mae_rtl)
    plt.title("Experiment 3 : DT/RT Learner-MAE")
    plt.xlabel("leaf size")
    plt.ylabel("MAE")
    plt.legend(['DTLearner', 'RTLearner'])
    plt.savefig("images/Figure 5.png")
    plt.close()


if __name__ == "__main__":

    if len(sys.argv) != 2:  # check if there are 2 args
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    if sys.argv[1] != "Data/Istanbul.csv":  # check if is correct file
        print("Wrong filename")
        sys.exit(1)
    inf = open(sys.argv[1])
    header = inf.readline()
    data = np.array([list(map(float, s.strip().split(",")[1:])) for s in inf.readlines()])

    #data = np.genfromtxt("Data/Istanbul.csv", delimiter=',')
    #data = np.delete(data, 0, axis=0)  # delete first row
    #data = np.delete(data, 0, axis=1)  # delete first column
    np.random.shuffle(data)  # Data provided must be randomly selected

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    exp1(train_x, train_y, test_x, test_y)
    exp2(train_x, train_y, test_x, test_y)
    exp2_2(train_x, train_y, test_x, test_y)
    exp3(train_x, train_y, test_x, test_y)

    # create a learner and train it
    # learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    # learner.add_evidence(train_x, train_y)  # train it
    # print(learner.author())

