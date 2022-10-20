import keras as tk
from keras.preprocessing import image
import numpy as np
import PIL
import tensorflow as tf

## load img
testimg = tk.utils.load_img("two.png")
testimg= tk.utils.img_to_array(testimg)

## change dims
testimg= np.expand_dims(testimg,0)
testimg= np.expand_dims(testimg,-1)

testimg = testimg[:,:,:,0,:]

#load saved tf model
loaded_mod = tk.models.load_model('lior_modelnew211.h5')

#predict for two.png
vcm_pre = loaded_mod.predict(testimg)

print(vcm_pre.argmax())



#load saved tflite model

interpreter = tf.lite.Interpreter('lior_modelnew211lite.h5')

input_details = interpreter.get_input_details()
out_details = interpreter.get_output_details()

interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], testimg)
interpreter.invoke()
tflite_pred_res= interpreter.get_tensor(out_details[0]['index'])
print(tflite_pred_res.argmax())