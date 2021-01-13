#Training NN to categorize articles by topic using Reuters dataset which has 46 different labeled topics

from keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)

#Create tensors just like in imdb
import numpy as np
def makeTensor(sequences, dimension=10000):
    output = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        output[i, sequence] = 1.
    return output

x_train = makeTensor(train_data)
x_test = makeTensor(test_data)

#This time labels need to be done differently
#Using one-hot encoding to put a 1 in the index corresponding to the index of the article's topic
#Algo essentially same as for data, but with smaller dimension because we only have 46 topics
def oneHot(labels, dimension=46):
    output = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        output[i, label] = 1.
    return output

y_train = oneHot(train_labels)
y_test = oneHot(test_labels)
#Keras has to_categorical() built in which does this for you

#Create model
from keras import models
from keras import layers

model = models.Sequential()
#Use 64 hidden units this time instead of 16 because we are now dealing with 46 topics instead of just binary classification
model.add(layers.Dense(64, activation = "relu", input_shape = (10000,)))
model.add(layers.Dense(64, activation = "relu"))
model.add(layers.Dense(46, activation = "softmax")) #Softmax assigns a probability distrubution over each of the 46 categories

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#Using categorical rather than binary crossentropy because we are now dealing with multiple categories

#Just using 1000 samples for validation set because training set is a bit smaller this time
x_val = x_train[:1000]
x_train_subset = x_train[1000:]

y_val = y_train[:1000]
y_train_subset = y_train[1000:]

#Train model
history = model.fit(x_train_subset, y_train_subset, epochs = 12, batch_size = 512, validation_data = (x_val, y_val))

#Plot model success
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1,len(loss)+1)

plt.plot(epochs, loss, 'bo', label = 'Training Set')
plt.plot(epochs, val_loss, 'b', label = 'Validation Set')
plt.title('Training and Validation Loss')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()

plt.savefig('Fig1')

plt.clf()
plt.plot(epochs, acc, 'bo', label = 'Training Set')
plt.plot(epochs, val_acc, 'b', label = 'Validation Set')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()

plt.savefig('Fig2')