# To Capture Frame
import cv2
import tensorflow as tf
# To process image array
import numpy as np


# import the tensorflow modules and load the model



# Attaching Cam indexed as 0, with the application software
mymodel=tf.keras.models.load_model('keras_model.h5')
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)
		
		
		
		#resize the frame
		resize_frame=cv2.resize(frame,(224,224))
		resize_frame=np.expand_dims(resize_frame,axis=0)
		# expand the dimensions
		
		# normalize it before feeding to the model
		resize_frame=resize_frame/255
		# get predictions from the model
		prediction=mymodel.predict(resize_frame)
		#print(prediction)
		rock=int(prediction[0][0]*100)
		paper=int(prediction[0][1]*100)
		scissor=int(prediction[0][2]*100)
		print(f'Rock: {rock}% ,Paper: {paper}%, scissor: {scissor}%')
		
		
		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
