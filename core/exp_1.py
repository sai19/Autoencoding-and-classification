from __future__ import division
from keras.layers import LSTM,Input,Lambda,Dense,Activation,Dropout
from keras import backend as K
from keras.models import Model,Sequential
import keras
from keras import metrics
from keras.datasets import mnist
import numpy as np
import cv2
from tqdm import tqdm
from keras.optimizers import RMSprop
import pickle
import random

batch_size = 128
num_classes = 10
epochs = 20

(x_train, y_train), (x_test, y_test) = mnist.load_data()


for i in tqdm(range(60000)):
	random.seed(i)
	if random.uniform(0,1)>-0.1:
		if y_train[i] == 0:
			x_train[i,2:7,0:5] = 255
		elif y_train[i] == 1:
			x_train[i,12:17,0:5] = 255
		elif y_train[i] == 2:
			x_train[i,22:27,0:5] = 255
		elif y_train[i] == 3:
			x_train[i,23:28,2:7] = 255
		elif y_train[i] == 4:
			x_train[i,23:28,12:17] = 255
		elif y_train[i] == 5:
			x_train[i,23:28,22:27] = 255
		elif y_train[i] == 6:
			x_train[i,0:5,2:7] = 255
		elif y_train[i] == 7:
			x_train[i,0:5,12:17] = 255
		elif y_train[i] == 8:
			x_train[i,2:7,23:28] = 255
		elif y_train[i] == 9:
			x_train[i,12:17,23:28] = 255									
#x_train = x_train/255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
'''
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.2)
score = model.evaluate(x_test, y_test, verbose=0)

'''
input_img = Input(shape=(784,))
encoded = Dense(512, activation='relu')(input_img)
encoded = Dropout(0.2)(encoded)
encoded = Dense(512, activation='relu')(encoded)
encoded = Dropout(0.2)(encoded)
decoded = Dense(784, activation='sigmoid')(encoded)
out = Dense(10,activation="softmax",name="final")(encoded)
model = Model(input_img, [decoded,out])
model.summary()
model.compile(optimizer='rmsprop', 
              loss=['binary_crossentropy', 'categorical_crossentropy'],
              loss_weights=[1.0, 1.0],metrics=["accuracy"])
history = model.fit(x_train, [x_train,y_train],
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_split=0.2)
score = model.evaluate(x_test, [x_test,y_test], verbose=0)
#'''
print(score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#for i in range(20):
pred = model.predict(x_test[60:80,:])
pred = pred[0]
pred = pred.reshape((20,28,28))*255
test = x_test[60:80,:].reshape((20,28,28))*255
for i in range(20):
	cv2.imshow("im",cv2.resize(test[i,:,:],(280,280)))
	cv2.imshow("img",cv2.resize(pred[i,:,:],(280,280)))
	cv2.waitKey(0)
print(history.history.keys())
with open('out_auto.pkl', 'wb') as f:
        pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)