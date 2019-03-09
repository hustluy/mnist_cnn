
import tensorflow as tf
import numpy as np

import input_mnist

train_data_file = './data/train-images-idx3-ubyte'
train_label_file = './data/train-labels-idx1-ubyte'

test_data_file = './data/t10k-images-idx3-ubyte'
test_label_file = './data/t10k-labels-idx1-ubyte'

x_train, y_train = input_mnist.load(train_data_file, train_label_file)
x_test, y_test = input_mnist.load(test_data_file, test_label_file)

x_train = x_train / 255.0
x_test = x_test / 255.0

cnn_model = tf.keras.Sequential()

cnn_model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3)))
cnn_model.add(tf.keras.layers.MaxPool2D(pool_size = (2, 2)))
cnn_model.add(tf.keras.layers.Flatten())
cnn_model.add(tf.keras.layers.Dense(units = 256, activation = tf.keras.activations.relu))
cnn_model.add(tf.keras.layers.Dense(units = 10, activation = tf.keras.activations.softmax))

cnn_model.compile(optimizer = tf.keras.optimizers.Adam(),
loss = tf.keras.losses.sparse_categorical_crossentropy,
metrics = [tf.keras.metrics.sparse_categorical_accuracy])

cnn_model.fit(x_train, y_train, batch_size = 16, epochs = 10)

print cnn_model.evaluate(x_test, y_test)
