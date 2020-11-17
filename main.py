from loadData import dataPreProcessing
from network import trainModel

        




train_X,valid_X,train_label,valid_label, test_X, train_X_one_hot, test_Y_one_hot = dataPreProcessing()
trainModel(train_X, train_label, valid_X, valid_label, test_X, train_X_one_hot, test_Y_one_hot)
