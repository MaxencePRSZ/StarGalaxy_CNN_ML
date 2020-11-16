import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

from loadData import createTestingData, createTrainingData

        

train_X, train_Y = createTestingData()


print('Training data shape : ', train_X.shape, train_Y.shape)

classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)