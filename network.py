import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from loadData import IMGSIZE
import matplotlib.pyplot as plt



#########################################################
####################  Parameters  #######################
#########################################################

BACTH_SIZE = 64
EPOCHS = 20
NUM_CLASSES = 2


def createModel():
    #First, we initialize our NN model
    fashion_model = Sequential()

    #Then, we feed him with different layers
    #First layer is a 32-3x3 Convolutional layer followed by a LeakyRELU and pooling
    fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(IMGSIZE,IMGSIZE,1),padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D((2, 2),padding='same'))
    fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))                  
    fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    fashion_model.add(Flatten())
    fashion_model.add(Dense(128, activation='linear'))
    fashion_model.add(LeakyReLU(alpha=0.1))                  
    fashion_model.add(Dense(NUM_CLASSES, activation='softmax'))
    fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    return fashion_model



def trainModel(train_X, train_label, valid_X, valid_label, test_X, train_X_one_hot, test_Y_one_hot):
    fashion_model = createModel()
    fashion_train = fashion_model.fit(train_X, train_label, batch_size=BACTH_SIZE,epochs=EPOCHS,verbose=1, shuffle=True, validation_data=(valid_X, valid_label))

    evaluateModel(fashion_train)

    test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])


def evaluateModel(fashion_train):
    accuracy = fashion_train.history['accuracy']
    val_accuracy = fashion_train.history['val_accuracy']
    loss = fashion_train.history['loss']
    val_loss = fashion_train.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()