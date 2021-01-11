#Training a NN to classify IMDB reviews as positive or negative in Keras

#Works by taking in data on the most used 10000 words, seeing what words are present in a given review,
#and then training a model to classify reviews as positive or negative based on reviews in a training set.

#Import IMDB dataset, already present in Keras
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

import numpy as np

#One-hot code data into tensor form
def makeTensor(sequences, dimension=10000):
    output = np.zeros((len(sequences), dimension))
    #i is the number of the seqeunce
    #sequence gives the words that do appear in a review as a binary vector
    for i, sequence in enumerate(sequences):
        output[i, sequence] = 1
    return output

tensor_train = makeTensor(train_data)
tensor_test = makeTensor(test_data)

#Turn labels into arrays for ease of use
label_train = np.asarray(train_labels).astype('float32')
label_test = np.asarray(test_labels).astype('float32')

#NN will have 2 intermediate layers with 16 hidden units each (using relu to override linear transformations only)
#and one layer which provides a scalar output
#relu also keeps >0 which is useful in this case
from keras import models
from keras import layers

#Sequential used for creating models with layers stacked sequentially on top of one another
model = models.Sequential()

#Add dense layers to model
model.add(layers.Dense(16, activation = 'relu', input_shape = (10000,))) #inputs are 10000 length vectors
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid')) #sigmoid to output probability between 0-1 of positive review

#Validation set of 10000 reviews, split up training set
tensor_val = tensor_train[:10000]
partial_tensor_train = tensor_train[10000:]

label_val = label_train[:10000]
partial_label_train = label_train[10000:]

#Training model

#Now need loss function, using binary_crossentropy because we're dealing with probabilities
#Optimizer is rmsprop
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Use history to see loss optimization
history = model.fit(partial_tensor_train, partial_label_train, epochs = 5, batch_size = 512, validation_data = (tensor_val, label_val))

#Graph the loss function
import matplotlib.pyplot as grapher

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, 6)

grapher.plot(epochs, loss_values, 'bo', label = "Training Set")
grapher.plot(epochs, val_loss_values, 'b', label = "Validation Set")

grapher.xlabel('Epochs')
grapher.ylabel('Loss')
grapher.legend()

grapher.savefig('Figure1')

#Graph accuracy
grapher.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

grapher.plot(epochs, acc_values, 'bo', label = "Training Set")
grapher.plot(epochs, val_acc_values, 'b', label = "Validation Set")

grapher.xlabel('Epochs')
grapher.ylabel('Accuracy')
grapher.legend()

grapher.savefig('Figure2')
