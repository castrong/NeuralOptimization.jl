import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
label_to_points = {} # group label to training point indices
for i, label in enumerate(y_train):
    if label not in label_to_points:
        label_to_points[label] = []
    label_to_points[label].append(i)

label = 2
index = 3
X = x_train[label_to_points[label][index]]
print("A training point with label {}".format(label), X)
plt.imshow(X, cmap='Greys')

X_norm = X.flatten() / 255
print("Normalized input: ", X_norm)
plt.imshow(X, cmap='Greys')
