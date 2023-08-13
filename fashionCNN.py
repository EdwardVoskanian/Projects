# Python 2/3 compatibility.
# from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_datasets as tfds

import math
import numpy as np
import matplotlib.pyplot as plt

import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm

# print(tf.__version__)

#tf.enable_eager_execution()


# Load the data set, fashion_mnist.
dataset,metadata = tfds.load("fashion_mnist",as_supervised = True,with_info = True)
train_dataset,test_dataset = dataset["train"],dataset["test"]

# Create class names
class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal",
               "Shirt","Sneaker","Bag","Ankle boot"]

# Check the number of training examples and test examples.
num_train_examples = metadata.splits["train"].num_examples
num_test_examples = metadata.splits["test"].num_examples
print("{}".format(num_train_examples))
print("{}".format(num_test_examples))

# This function will normalize the images.
def normalize(images,labels):
    images = tf.cast(images,tf.float32)
    images /= 255
    return images,labels
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),padding = 'same',activation = tf.nn.relu,
                           input_shape = (28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2),strides = 2),
    tf.keras.layers.Conv2D(64,(3,3),padding = 'same',activation = tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2,2),strides = 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation = tf.nn.relu),
    tf.keras.layers.Dense(10,activation = tf.nn.softmax)
    ])

# Compile the model.
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# Train the model.
BATCH_SIZE = 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
model.fit(train_dataset,epochs = 10,steps_per_epoch = math.ceil(num_train_examples/32))

# Check for accuracy.
test_loss,test_accuracy = model.evaluate(test_dataset,steps = math.ceil(num_test_examples/BATCH_SIZE)) 










   