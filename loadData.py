import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random



CATEGORIES = ["star", "galaxy"]
IMGSIZE = 512



def createTrainingData():
    #Directory of the training dataset
    DATADIR = "../archive/data/train/"
    training_data = []
    train_X, train_Y = [], []

    #Going through different category of data
    for cat in CATEGORIES:
        #Selecting folder betwenen categories
        path = os.path.join(DATADIR, cat)
        #getting index of each category
        classnum = CATEGORIES.index(cat)
        #going through each image
        for img in os.listdir(path)[:100]:
            try:
                #Reading and giving a gray scale
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                #resize to the size given
                new_array = cv2.resize(img_array, (IMGSIZE, IMGSIZE))
                training_data.append([new_array, classnum])
            except Exception as e:
                print("Image {} from {} is not working properly".format(img, cat))
        
    #Shuffle data so the first half is not the first category
    random.shuffle(training_data)

    #split the image into train_X and the classification into train_Y
    for x, y in training_data:
        train_X.append(x)
        train_Y.append(y)

    #change to numpy array
    train_X = np.array(train_X)
    train_Y= np.array(train_Y)

    #Useless but classy
    training_data.clear()

    return train_X, train_Y


def createTestingData():
    #Directory of the testing dataset
    DATADIR = "../archive/data/validation/"
    testing_data = []
    test_X, test_Y = [], []

    #Going through different category of data
    for cat in CATEGORIES:
        #Selecting folder betwenen categories
        path = os.path.join(DATADIR, cat)
        #getting index of each category
        classnum = CATEGORIES.index(cat)
        #going through each image
        for img in os.listdir(path)[:100]:
            try:
                #Reading and giving a gray scale
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                #resize to the size given
                new_array = cv2.resize(img_array, (IMGSIZE, IMGSIZE))
                testing_data.append([new_array, classnum])
            except Exception as e:
                print("Image {} from {} is not working properly".format(img, cat))
        
    #Shuffle data so the first half is not the first category
    random.shuffle(testing_data)

    #split the image into test_X and the classification into test_Y
    for x, y in testing_data:
        test_X.append(x)
        test_Y.append(y)

    #change to numpy array
    test_X = np.array(test_X)
    test_Y= np.array(test_Y)

    #Useless but classy
    testing_data.clear()

    return test_X, test_Y