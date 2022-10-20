import numpy as np
import PIL
import utils
import cv2
import tflite_runtime.interpreter as tflite



##load image
testimg = cv2.imread("two.png")
testimg = np.array(testimg,dtype='float32')

## change dims
testimg= np.expand_dims(testimg,0)
testimg= np.expand_dims(testimg,-1)
testimg = testimg[:,:,:,0,:]


#load saved tflite model
interpreter = tflite.Interpreter('lior_modelnew211lite.h5')
input_details = interpreter.get_input_details()
out_details = interpreter.get_output_details()
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], testimg)
interpreter.invoke()
tflite_pred_res= interpreter.get_tensor(out_details[0]['index'])
print(tflite_pred_res.argmax())
