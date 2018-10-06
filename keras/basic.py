import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import np_utils
from keras.models import load_model
import matplotlib.pyplot as plt




seed = 7
np.random.seed(7)
(x_train, y_train), (x_test, y_test) = mnist.load_data();

'''plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
plt.show()
print y_train[0]'''

num_pixels = x_train.shape[1]*x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')
x_train = x_train/255
x_test = x_test/255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
'''
def base_line():
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


print len(x_test[1])	

model = base_line()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200)
score = model.evaluate(x_test, y_test)
print score


print "saving model..."

model.save('model_1')
'''

new_model = load_model('model_1')
prediction = new_model.predict([x_test])
print np.argmax(prediction[0])





