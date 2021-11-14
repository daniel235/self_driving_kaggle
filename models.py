import tensorflow as tf

from tensorflow.keras import layers, models 
import matplotlib.pyplot as plt


class Models:
    def __init__(self):
        self.type = "cpu"
        self.train = []
        self.test = []
        self.typeCheck()
        self.history = None
        self.accuracy = None


    def typeCheck(self):
        self.type = "cpu"


    def cnn(self):
        #size of input 3 * 160 * 320 * 1
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(160, 320, 1)))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64, (3,3), activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64, (3,3), activation='relu'))
        print(model.summary())
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10))
        print(model.summary())
        return model


    def neuralNet(self):
        pass


    #ensemble learning
    def deep_learning(self):
        #size
        pass


    def optimization(self, model, optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)):
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        return model


    def train(self, model, train, test):
        self.history = model.fit(train[0], train[1], epochs=10, validation_data=(test[0], test[1]))
        test_loss, test_acc = model.evaluate(test[0], test[1], verbose=2)
        self.accuracy = test_acc
        print("Accuracy ", self.accuracy)


    def plot(self, model):
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')

    