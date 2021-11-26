import numpy as np
import pandas as pd

def read_data(path, target_column):
    data = pd.read_csv(path)
    X = data.iloc[:, :-1]
    Y = data[target_column]

    return X, Y

def suffle_data (X, Y):
    X['target'] = Y
    X = X.sample(frac=1)

    return X.iloc[:, :-1], X.iloc[:, -1]



def split_data(X, Y, ratio):
    N = Y.count()
    p = int(ratio*N)

    x_train = X.iloc[0:p, :]
    y_train = Y.iloc[0:p]
    x_test = X.iloc[p:, :]
    y_test = Y.iloc[p:]

    return x_train, y_train, x_test, y_test


def precision (Y, predict):

    return (Y == predict).sum()/Y.count()

def confusion_matrix (Y, predict):
    TP = TN = FP = FN = 0

    for i in range(Y.count()):
        if Y.iloc[i] == 1 and predict[i] == 1:
            TP += 1
        elif Y.iloc[i] == 0 and predict[i] == 0:
            TN += 1
        elif Y.iloc[i] == 0 and predict[i] == 1:
            FP += 1
        elif Y.iloc[i] == 1 and predict[i] == 0:
            FN += 1

    confusin_matrix = [[TP, FP], [FN, TN]]
    return confusin_matrix

def report (Y, predict):
    confusio_matrix = confusion_matrix(Y, predict)
    TP, FP = confusio_matrix[0]
    FN, TN = confusio_matrix[1]

    Accuracy = (TP + TN)/(TP + FP + FN + TN)
    Precision = TP/(TP + FP)
    Recall = TP/(TP + FN)
    F1score = (2*Precision*Recall)/(Precision + Recall)

    return Accuracy, Precision, Recall, F1score