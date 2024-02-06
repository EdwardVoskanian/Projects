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

# We can plot some of our images. Here, we look at 25 images with labels. 
plt.figure(figsize = (10,10))
i = 0
for image,label in test_dataset.take(25):
    image = image.numpy().reshape((28,28))
    plt.subplot(5,5,i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image,cmap = plt.cm.binary)
    plt.xlabel(class_names[label])
    i += 1

# Define the model.
l0 = tf.keras.layers.Flatten(input_shape = (28,28,1))
l1 = tf.keras.layers.Dense(128,activation = tf.nn.relu)
l2 = tf.keras.layers.Dense(10,activation = tf.nn.softmax)
model = tf.keras.Sequential([l0,l1,l2])

# Complile the model.
model.compile(optimizer = "adam",
              loss = "sparse_categorical_crossentropy",
              metrics = "accuracy")

# Train the model.
BATCH_SIZE = 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
model.fit(train_dataset,epochs = 5,steps_per_epoch = math.ceil(num_train_examples/32))

# Check for accuracy.
test_loss,test_accuracy = model.evaluate(test_dataset,steps = math.ceil(num_test_examples/BATCH_SIZE))    

# We can make predictions
for test_images,test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)


