
# -*- coding: utf-8 -*-
# @Time : 20-7-20
# @Author : zhangT
# @Company : *
# @File : test.py
import numpy as np
import math,cv2,os,sys
sys.path.append("src/")
from Model import Landmark_model, RGBAnti, Eye_model
# np.set_printoptions(threshold=np.inf)

def process(img, box):
	
	landmark_model = Landmark_model()
	eye_model = Eye_model()
	rgbanti_model = RGBAnti()
	x1,y1,x2,y2 = box
	w_box = x2-x1
	rate = 0.1
	x1 = int(x1+w_box*rate)
	x2 = int(x2-w_box*rate)
	crop_face = img[y1:y2, x1:x2]
	start = cv2.getTickCount()
	confidence, landmark_box, landmark = landmark_model.forward(img.copy(), box)
	# eye
	lf_eye_status, rg_eye_status = eye_model.forward(img.copy(), box, landmark)
	# anti
	anti_status = rgbanti_model.forward(crop_face)
	# occl
	end = cv2.getTickCount()
	during1 = (end - start) / cv2.getTickFrequency()
	print('compute face waste time ', during1*1000)	
	# draw
	points_x,points_y = landmark
	lf_out = np.argmax(lf_eye_status[0])
	rg_out = np.argmax(rg_eye_status[0])
	eye_info = 'open'
	close_score = 0.0
	if lf_out==2 and rg_out == 2:
		eye_info = 'close'
		close_score = (lf_eye_status[0][2]+rg_eye_status[0][2])/2
	info = 'eye_close: '+str(eye_info)+' '+str(close_score)[:5]
	reality_face = 0
	text_colour = (0,0,255)
	if anti_status[0][2] > 0.6:
		reality_face = 1
		text_colour = (0,255,0)
	rgb_real = anti_status[0][2]
	start_x = 80
	start_y = 20
	anti_info = "anti: "+str("real" if reality_face==1 else "fake")
	cv2.putText(img,info,(start_x,start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,text_colour,1)
	cv2.putText(img,anti_info,(start_x,start_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,text_colour,1)
	for i in range(5):
		cv2.circle(img,(int(points_x[i]),int(points_y[i])),2,text_colour,1)
	cv2.rectangle(img, (x1, y1), (x2, y2), text_colour, 2)
	return 0








if __name__ == '__main__':
	# detect to img
	face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
	path = "img/"
	img_list = os.listdir(path)
	for i in range(len(img_list)):
		img = cv2.imread(path+img_list[i])
		start = cv2.getTickCount()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# detect
		faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.15, minNeighbors = 5, minSize = (5, 5))
		end = cv2.getTickCount()
		during1 = (end - start) / cv2.getTickFrequency()
		print('detect waste time ', during1*1000)
		for(x, y, w, h) in faces:
			box = np.array([x,y,x+w,y+h])
			# compute face
			real_count = process(img, box)
		cv2.imwrite(img_list[i], img)
		cv2.imshow('', img)
		cv2.waitKey(0)
	