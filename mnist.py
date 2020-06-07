import tensorflow as tf
import keras as k
import matplotlib.pyplot as plt
import numpy as np

#importing dataset
mnist= tf.keras.datasets.mnist
(x_train,y_train), (x_test,y_test)= mnist.load_data('pythonis.whl')
x_train=k.utils.normalize(x_train,axis=1)
x_test=k.utils.normalize(x_test,axis=1)
plt.imshow(x_test[100])
plt.show()
x_train=x_train.reshape((x_train.shape[0],28,28,1)).astype('float32')
x_test=x_test.reshape((x_test.shape[0],28,28,1)).astype('float32')

#building cnn model
model=k.models.Sequential()
model.add(k.layers.Flatten())
model.add(k.layers.Dense(128,activation="relu"))
model.add(k.layers.Dense(128,activation="relu"))
model.add(k.layers.Dense(10,activation="softmax"))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#model training
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=5,verbose=2)
#model accuracy and loss
val_loss,val_acc= model.evaluate(x_test,y_test)


#prediction of new numbers
new_number=model.predict(x_test)
print('The given number is:')
print(np.argmax(new_number[100]))