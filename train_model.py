import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
# from keras.utils import np.utils

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the images to values between 0 and 1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape to include the depth dimension (for CNN)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense





# Compute distribution for training and test sets
train_counts = np.bincount(y_train)
test_counts = np.bincount(y_test)

print("Training set distribution:", train_counts)
print("Test set distribution:", test_counts)

# Plot the distribution for the training set
plt.figure(figsize=(10,5))
plt.bar(range(10), train_counts)
plt.title("No of images for each class in train set")
plt.xlabel("Class Id")
plt.ylabel("Number of images")
plt.show()
#######


model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
