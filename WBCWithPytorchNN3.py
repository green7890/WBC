
# this file should give complete output, run with WBCVenv3_13
# based on https://www.tensorflow.org/tutorials/quickstart/beginner

import tensorflow as tf
import numpy as np
import math


class TFNeuralNetwork:
    def __init__(self,input_shape=(10), output_length=10):
        self.model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(output_length)
        ])

        # predictions = self.model(x_train[:1]).numpy()
        # print(predictions)

        # print(tf.nn.softmax(predictions).numpy())

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # print(self.loss_fn(y_train[:1], predictions).numpy())

        self.model.compile(optimizer='adam',
                    loss=self.loss_fn,
                    metrics=['accuracy'])

    def train(self, x_train, y_train):
        n_epochs = 5 # 1 (reduced from 5 to 1 to make several water through pipes mutation development faster to get to the later steps)
        self.model.fit(x_train, y_train, epochs=n_epochs)
    def predict(self, x_test,  y_test):
        self.model.evaluate(x_test,  y_test, verbose=2)

        probability_model = tf.keras.Sequential([
        self.model,
        tf.keras.layers.Softmax()
        ])

        print(probability_model(x_test[:5]))


if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0


    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
    # (60000,) means a numpy array of one dimension
    # print(type(x_train), type(y_train), type(x_test), type(y_test))
    # <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'>

    tfnn = TFNeuralNetwork(input_shape=x_train.shape[1:3], output_length=10)
    tfnn.train(x_train, y_train)
    tfnn.predict(x_test, y_test)

    print(y_train)

    rng = np.random.default_rng()
    
    # mutation 1: use random numpy instead of mnist
    x_train2 = rng.random(size=(60000,28,28))
    y_train2 = rng.integers(0, 10, size=(60000))
    print("y_train2 max min shape", np.max(y_train2), np.min(y_train2), y_train2.shape) 
    x_test2 = rng.random(size=(10000,28,28))
    y_test2 = rng.integers(0, 10, size=(10000))
    print("y_test2 max min shape", np.max(y_test2), np.min(y_test2), y_test2.shape)
   
    tfnn2 = TFNeuralNetwork(input_shape=x_train2.shape[1:3], output_length=10)
    tfnn2.train(x_train2, y_train2)
    tfnn2.predict(x_test2, y_test2)

    # mutation 2: use input dim (30,) instead of (28, 28)
    x_train3 = rng.random(size=(60000,30))
    x_test3 = rng.random(size=(10000,30))
    shape3 = x_train3.shape[1:2]
    print(shape3)
    tfnn3 = TFNeuralNetwork(input_shape=shape3, output_length=10)
    tfnn3.train(x_train3, y_train2)
    tfnn3.predict(x_test3, y_test2)

    # mutation 3: use output  size 2 instead of 10
    y_train4 = rng.integers(0, 2, size=(60000))
    print("y_train2 max min shape", np.max(y_train4), np.min(y_train4), y_train4.shape)
    y_test4 = rng.integers(0, 2, size=(10000))
    print("y_test2 max min shape", np.max(y_test4), np.min(y_test4), y_test4.shape)

    print("xtrain type",type(x_train3), type(x_train3[0]), type(x_train3[0][0]))
    print("ytrain type", type(y_train4), type(y_train4[0]))

    tfnn4 = TFNeuralNetwork(input_shape=shape3, output_length=2)
    tfnn4.train(x_train3, y_train4)
    tfnn4.predict(x_test3, y_test4)


### working tutorial code:

# import tensorflow as tf
# print("TensorFlow version:", tf.__version__)

# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10)
# ])

# predictions = model(x_train[:1]).numpy()
# print(predictions)

# tf.nn.softmax(predictions).numpy()

# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# print(loss_fn(y_train[:1], predictions).numpy())

# model.compile(optimizer='adam',
#               loss=loss_fn,
#               metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=5)

# model.evaluate(x_test,  y_test, verbose=2)

# probability_model = tf.keras.Sequential([
#   model,
#   tf.keras.layers.Softmax()
# ])

# print(probability_model(x_test[:5]))