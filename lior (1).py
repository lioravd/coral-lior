# -*- coding: utf-8 -*-

# -- Sheet --

import numpy as np
import tensorflow.keras as tk
from matplotlib import pyplot
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing import image
import os
import keras

mnist=tk.datasets.mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()



train_X = np.expand_dims(train_X,-1)
test_X = np.expand_dims(test_X,-1)



train_y= tk.utils.to_categorical(train_y,num_classes=None)
test_y= tk.utils.to_categorical(test_y,num_classes=None)




vcm = tk.Sequential()
vcm.add(Conv2D(32,(3,3),input_shape = (28,28,1), activation= 'relu'))

vcm.add(MaxPooling2D(pool_size = (2,2)))

vcm.add(Conv2D(64,(3,3),activation='relu' ))
vcm.add(MaxPooling2D(pool_size = (2,2)))

vcm.add(Conv2D(128,(3,3),activation='relu' ))
vcm.add(MaxPooling2D(pool_size = (2,2)))

#vcm.add(Conv2D(256,(3,3),activation='relu' ))
#vcm.add(MaxPooling2D(pool_size = (2,2)))

vcm.add(Flatten())
vcm.add(Dense(units=1024,activation = 'relu'))
vcm.add(Dense(units=10,activation = 'softmax'))

vcm.compile(optimizer='adam', loss = 'categorical_crossentropy' , metrics=['accuracy'])
vcm.summary()

history = vcm.fit(train_X,train_y,epochs=15,validation_data=(test_X,test_y))

vcm.save("lior_modelnew211.h5")
vcm.save_weights("lior_weightsnew211.hdf5")





loaded_mod=keras.models.load_model('lior_modelnew211.h5')

vcm_pre = vcm.predict(test_X)
print(vcm_pre)







testimg = image.load_img("three.png")
testimg= image.img_to_array(testimg)
pyplot.imshow(testimg)

testimg=np.expand_dims(testimg,0)

testimg=np.expand_dims(testimg,-1)

print(testimg[:,:,:,0,:].shape)

pre_result = vcm.predict(testimg[:,:,:,0,:])
print("pre_result: "  , pre_result[0])
print(pre_result[0].argmax())

converter = tf.lite.TFLiteConverter.from_keras_model(loaded_mod)
tflite1_model22 = converter.convert()

open('lior_modelnew211lite.h5','wb').write(tflite1_model22)

print("main model size : " , os.path.getsize("lior_modelnew211.h5")/(1024*1024))
print("Lite model size : " , os.path.getsize("lior_weightsnew211.hdf5")/(1024*1024))

interpreter = tf.lite.Interpreter('lior_modelnew211lite.h5')

input_details = interpreter.get_input_details()
out_details = interpreter.get_output_details()


print(input_details)
print("-"*10)
print(input_details)




interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], testimg[:,:,:,0,:])
interpreter.invoke()

tflite_pred_res= interpreter.get_tensor(out_details[0]['index'])
print(tflite_pred_res.argmax())



