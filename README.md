# StarGalaxy_CNN_ML

## Description of the Project

The purpose of the project here is to determine what type of celestial star we are looking at on a large set of images. We want to automatize the classification of these celestial objects and thus allow astronomers to save time on redundant tasks. Since the classification will happen on images, we will need to use a Convolutional Neural Network (CNN) also called ConvNet to compute and classify the image of stars.

## Run the project

To run the project, you first have to download the dataset from here : https://www.kaggle.com/siddharthchaini/stars-and-galaxies

You will have to specify the url of the previously downloaded dataset in the loadData.py files at the lines 19 and 60 :

    def createTrainingData():
      DATADIR = "../archive/data/train/"
      [...]

    def createTestingData():
      DATADIR = "../archive/data/validation/"
      [...]

You will also have to install all the python libraries with this command in a cmd at the root of the directory :

    pip install -R requirements.txt
