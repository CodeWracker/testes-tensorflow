import tensorflow as tf
from tensorflow import keras
import numpy as np


(a,b),(c,d) = keras.datasets.fashion_mnist.load_data()
#tf.data.Dataset.
train_data = tf.constant([[[10,5]],[[12,6]],[[3,1]],[[53,5]],[[5,15]]])
train_labels = tf.constant([0,1,1,0,1])
test_data = tf.constant([[[12,3]],[[1,30]],[[41,30]]])
test_labels = tf.constant([0,1,1])

print(d.shape)
print(train_labels.shape)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1,2)),
    keras.layers.Dense(2,activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=["accuracy"])

model.fit(train_data, train_labels, epochs=100)
test_loss,test_acc = model.evaluate(test_data,test_labels)
print("Test Acc: ", test_acc)
input_test1 = int(input('1: '))
input_test2 = int(input('2: '))
input_data=[[input_test1,input_test2]]
input_tf = tf.constant(input_data)
prediction = model.predict([input_tf])
print(prediction)