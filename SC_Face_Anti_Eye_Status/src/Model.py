# -*- coding: utf-8 -*-
# @Time : 20-7-20
# @Author : zhangT
# @Company : *
# @File : Model.py
from base import Conv2D, Prelu, Relu, pooling, InnerProduct, Softmax
import numpy as np
import math,cv2,sys
sys.path.append("param/")
import params_landmark
import params_eye40
import params_rgbanti
import params_1occl56
# np.set_printoptions(threshold=np.inf)

class Landmark_model:
	def __init__(self):
		# params
		dict_h = params_landmark.dict_h
		# conv1
		mutil = 100000000
		data = dict_h['conv1_w']['data']
		shape_conv1w = dict_h['conv1_w']['shape']
		self.data_conv1w = np.array(data).reshape(shape_conv1w)/mutil
		data = dict_h['conv1_b']['data']
		shape_conv1b = dict_h['conv1_b']['shape']
		self.data_conv1b = np.array(data).reshape(shape_conv1b)/mutil
		# prelu1
		data = dict_h['prelu1_w']['data']
		shape_prelu1w = dict_h['prelu1_w']['shape']
		self.data_prelu1w = np.array(data).reshape(shape_prelu1w)/mutil
		# conv2
		data = dict_h['conv2_w']['data']
		shape_conv2w = dict_h['conv2_w']['shape']
		self.data_conv2w = np.array(data).reshape(shape_conv2w)/mutil
		data = dict_h['conv2_b']['data']
		shape_conv2b = dict_h['conv2_b']['shape']
		self.data_conv2b = np.array(data).reshape(shape_conv2b)/mutil
		# prelu2
		data = dict_h['prelu2_w']['data']
		shape_prelu2w = dict_h['prelu2_w']['shape']
		self.data_prelu2w = np.array(data).reshape(shape_prelu2w)/mutil
		# conv3
		data = dict_h['conv3_w']['data']
		shape_conv3w = dict_h['conv3_w']['shape']
		self.data_conv3w = np.array(data).reshape(shape_conv3w)/mutil
		data = dict_h['conv3_b']['data']
		shape_conv3b = dict_h['conv3_b']['shape']
		self.data_conv3b = np.array(data).reshape(shape_conv3b)/mutil
		# prelu3
		data = dict_h['prelu3_w']['data']
		shape_prelu3w = dict_h['prelu3_w']['shape']
		self.data_prelu3w = np.array(data).reshape(shape_prelu3w)/mutil
		# conv4
		data = dict_h['conv4_w']['data']
		shape_conv4w = dict_h['conv4_w']['shape']
		self.data_conv4w = np.array(data).reshape(shape_conv4w)/mutil
		data = dict_h['conv4_b']['data']
		shape_conv4b = dict_h['conv4_b']['shape']
		self.data_conv4b = np.array(data).reshape(shape_conv4b)/mutil
		# prelu4
		data = dict_h['prelu4_w']['data']
		shape_prelu4w = dict_h['prelu4_w']['shape']
		self.data_prelu4w = np.array(data).reshape(shape_prelu4w)/mutil
		# conv5
		data = dict_h['conv5_w']['data']
		shape_conv5w = dict_h['conv5_w']['shape']
		self.data_conv5w = np.array(data).reshape(shape_conv5w)/mutil
		data = dict_h['conv5_b']['data']
		shape_conv5b = dict_h['conv5_b']['shape']
		self.data_conv5b = np.array(data).reshape(shape_conv5b)/mutil
		# prelu5
		data = dict_h['prelu5_w']['data']
		shape_prelu5w = dict_h['prelu5_w']['shape']
		self.data_prelu5w = np.array(data).reshape(shape_prelu5w)/mutil
		# conv6-1_w
		data = dict_h['conv6-1_w']['data']
		shape_conv6_1w = dict_h['conv6-1_w']['shape']
		self.data_conv6_1w = np.array(data).reshape(shape_conv6_1w)/mutil
		data = dict_h['conv6-1_b']['data']
		shape_conv6_1b = dict_h['conv6-1_b']['shape']
		self.data_conv6_1b = np.array(data).reshape(shape_conv6_1b)/mutil
		# conv6-2_w
		data = dict_h['conv6-2_w']['data']
		shape_conv6_2w = dict_h['conv6-2_w']['shape']
		self.data_conv6_2w = np.array(data).reshape(shape_conv6_2w)/mutil
		data = dict_h['conv6-2_b']['data']
		shape_conv6_2b = dict_h['conv6-2_b']['shape']
		self.data_conv6_2b = np.array(data).reshape(shape_conv6_2b)/mutil
		# conv6-3_w
		data = dict_h['conv6-3_w']['data']
		shape_conv6_3w = dict_h['conv6-3_w']['shape']
		self.data_conv6_3w = np.array(data).reshape(shape_conv6_3w)/mutil
		data = dict_h['conv6-3_b']['data']
		shape_conv6_3b = dict_h['conv6-3_b']['shape']
		self.data_conv6_3b = np.array(data).reshape(shape_conv6_3b)/mutil

	def Padding(self, box, im_height, im_width):
		box[0] = np.maximum(0, box[0])
		box[1] = np.maximum(0, box[1])
		box[2] = np.minimum(im_height - 1, box[2])
		box[3] = np.minimum(im_width - 1, box[3])
		return box

	def BBox2Square(self, box):
		square_bbox = box.copy()
		w = box[2] - box[0] + 1
		h = box[3] - box[1] + 1
		max_side = np.maximum(h, w)
		square_bbox[0] = box[0] + (w - max_side) * 0.5
		square_bbox[1] = box[1] + (h - max_side) * 0.5
		square_bbox[2] = square_bbox[0] + max_side - 1
		square_bbox[3] = square_bbox[1] + max_side - 1
		return square_bbox

	def landmark_forward(self, input_data):
		# model_net
		# conv1
		start = cv2.getTickCount()
		input_data = input_data* 0.007843 - 1
		conv_out = Conv2D(input_data, input_data.shape,16,3,1,'VALID', self.data_conv1w, self.data_conv1b)
		conv_out = Prelu(conv_out, self.data_prelu1w)
		conv_out = pooling(conv_out, 3, 2, 1, 'max')
		conv_out = Conv2D(conv_out, conv_out.shape,32,3,1,'VALID', self.data_conv2w, self.data_conv2b)		
		conv_out = Prelu(conv_out, self.data_prelu2w)
		conv_out = pooling(conv_out, 3, 2, 0, 'max')
		# conv3
		conv_out = Conv2D(conv_out, conv_out.shape,32,3,1,'VALID', self.data_conv3w, self.data_conv3b)
		conv_out = Prelu(conv_out, self.data_prelu3w)
		conv_out = pooling(conv_out, 2, 2, 0, 'max')
		# conv4
		conv_out = Conv2D(conv_out, conv_out.shape,64,2,1,'VALID', self.data_conv4w, self.data_conv4b)
		conv_out = Prelu(conv_out, self.data_prelu4w)
		conv_out = InnerProduct(conv_out, self.data_conv5w, self.data_conv5b)
		conv_out = Prelu(conv_out, self.data_prelu5w)
		# InnerProduct
		confidence = InnerProduct(conv_out, self.data_conv6_1w, self.data_conv6_1b)
		softmax_out = Softmax(confidence)
		box = InnerProduct(conv_out, self.data_conv6_2w, self.data_conv6_2b)
		landmark = InnerProduct(conv_out, self.data_conv6_3w, self.data_conv6_3b)
		end = cv2.getTickCount()
		during1 = (end - start) / cv2.getTickFrequency()
		return confidence, box, landmark
	
	def gress(self, box, landmark, befor_box):
		x1,y1,x2,y2 = befor_box
		w_box = x2-x1
		h_box = y2-y1
		box[0][0] = box[0][0]*w_box+x1
		box[0][1] = box[0][1]*h_box+y1
		box[0][2] = box[0][2]*w_box+x1
		box[0][3] = box[0][3]*h_box+y1
		points_x = landmark[0][0:10:2]
		points_y = landmark[0][1:10:2]
		points_x = points_x * w_box + x1
		points_y = points_y * h_box + y1
		return box,[points_x, points_y]

	def forward(self, img, box):
		befor_box = box.copy()
		size = 48
		box = self.BBox2Square(box)
		box = self.Padding(box, img.shape[1], img.shape[0])
		x1 = int(box[0])
		y1 = int(box[1])
		x2 = int(box[2])
		y2 = int(box[3])
		
		im_crop = img[y1:y2, x1:x2, :]
		input_data = cv2.resize(im_crop, (size, size))
		input_data = np.transpose(input_data, (2,0,1))
		input_data = np.expand_dims(input_data, axis=0)
		confidence, box, landmark = self.landmark_forward(input_data)
		new_box , points = self.gress(box, landmark, befor_box)
		return confidence, new_box , points

class Eye_model:
	def __init__(self):
		# params
		dict_h = params_eye40.dict_h
		# conv1
		mutil = 100000000
		data = dict_h['conv1_w']['data']
		shape_conv1w = dict_h['conv1_w']['shape']
		self.data_conv1w = np.array(data).reshape(shape_conv1w)/mutil
		data = dict_h['conv1_b']['data']
		shape_conv1b = dict_h['conv1_b']['shape']
		self.data_conv1b = np.array(data).reshape(shape_conv1b)/mutil
		# prelu1
		data = dict_h['prelu1_w']['data']
		shape_prelu1w = dict_h['prelu1_w']['shape']
		self.data_prelu1w = np.array(data).reshape(shape_prelu1w)/mutil
		# conv2
		data = dict_h['conv2_w']['data']
		shape_conv2w = dict_h['conv2_w']['shape']
		self.data_conv2w = np.array(data).reshape(shape_conv2w)/mutil
		data = dict_h['conv2_b']['data']
		shape_conv2b = dict_h['conv2_b']['shape']
		self.data_conv2b = np.array(data).reshape(shape_conv2b)/mutil
		# prelu2
		data = dict_h['prelu2_w']['data']
		shape_prelu2w = dict_h['prelu2_w']['shape']
		self.data_prelu2w = np.array(data).reshape(shape_prelu2w)/mutil
		# conv4
		data = dict_h['conv4_w']['data']
		shape_conv4w = dict_h['conv4_w']['shape']
		self.data_conv4w = np.array(data).reshape(shape_conv4w)/mutil
		data = dict_h['conv4_b']['data']
		shape_conv4b = dict_h['conv4_b']['shape']
		self.data_conv4b = np.array(data).reshape(shape_conv4b)/mutil
	
		
	def Eye_forward(self, input_data):
		# model_net
		# conv1
		start = cv2.getTickCount()
		# conv1
		conv_out = Conv2D(input_data, input_data.shape,16,3,1,'VALID', self.data_conv1w, self.data_conv1b)
		# prelu
		conv_out = Prelu(conv_out, self.data_prelu1w)
		# pooling1
		conv_out = pooling(conv_out, 3, 2, 1, 'max')
		# conv2
		conv_out = Conv2D(conv_out, conv_out.shape,32,3,1,'VALID', self.data_conv2w, self.data_conv2b)
		# prelu
		conv_out = Prelu(conv_out, self.data_prelu2w)
		# pooling2
		conv_out = pooling(conv_out, 2, 2, 0, 'max')
		# inner
		conv_out = InnerProduct(conv_out, self.data_conv4w, self.data_conv4b)
		softmax_out = Softmax(conv_out)
		end = cv2.getTickCount()
		during1 = (end - start) / cv2.getTickFrequency()
		return softmax_out
	
	def forward(self, img_cp, box, landmark):
		x1,y1,x2,y2 = box
		w_box = x2-x1
		h_box = y2-y1
		eye_wh_half = h_box*0.12
		ud_shift = h_box*0.05
		left_ld = [landmark[0][0], landmark[1][0]]
		right_ld = [landmark[0][1], landmark[1][1]]
		y1 = int(left_ld[1]-eye_wh_half-ud_shift)
		h = int(2*(eye_wh_half+ud_shift))
		x1 = int(left_ld[0]-eye_wh_half)
		w = int(2*eye_wh_half)
		lf_im = img_cp[y1:y1+h, x1:x1+w]
		y1 = int(right_ld[1]-eye_wh_half-ud_shift)
		x1 = int(right_ld[0]-eye_wh_half)
		rg_im = img_cp[y1:y1+h, x1:x1+w]
		lf_im = cv2.resize(lf_im, (30,40))
		rg_im = cv2.resize(rg_im, (30,40))
		input_data = np.transpose(lf_im, (2,0,1))
		input_data = np.expand_dims(input_data, axis=0)
		lf_out = self.Eye_forward(input_data)
		input_data = np.transpose(rg_im, (2,0,1))
		input_data = np.expand_dims(input_data, axis=0)
		rg_out = self.Eye_forward(input_data)
		return lf_out, rg_out

class RGBAnti:
	def __init__(self):
		dict_h = params_rgbanti.dict_h
		# conv1-ft
		mutil = 100000000
		data = dict_h['conv1_w']['data']
		shape_conv1w = dict_h['conv1_w']['shape']
		self.data_conv1w = np.array(data).reshape(shape_conv1w)/mutil
		data = dict_h['conv1_b']['data']
		shape_conv1b = dict_h['conv1_b']['shape']
		self.data_conv1b = np.array(data).reshape(shape_conv1b)/mutil
		# prelu1
		data = dict_h['prelu1_w']['data']
		shape_prelu1w = dict_h['prelu1_w']['shape']
		self.data_prelu1w = np.array(data).reshape(shape_prelu1w)/mutil
		# conv2
		data = dict_h['conv2_w']['data']
		shape_conv2w = dict_h['conv2_w']['shape']
		self.data_conv2w = np.array(data).reshape(shape_conv2w)/mutil
		data = dict_h['conv2_b']['data']
		shape_conv2b = dict_h['conv2_b']['shape']
		self.data_conv2b = np.array(data).reshape(shape_conv2b)/mutil
		# prelu2
		data = dict_h['prelu2_w']['data']
		shape_prelu2w = dict_h['prelu2_w']['shape']
		self.data_prelu2w = np.array(data).reshape(shape_prelu2w)/mutil
		# conv3
		data = dict_h['conv3_w']['data']
		shape_conv3w = dict_h['conv3_w']['shape']
		self.data_conv3w = np.array(data).reshape(shape_conv3w)/mutil
		data = dict_h['conv3_b']['data']
		shape_conv3b = dict_h['conv3_b']['shape']
		self.data_conv3b = np.array(data).reshape(shape_conv3b)/mutil
		# prelu3
		data = dict_h['prelu3_w']['data']
		shape_prelu3w = dict_h['prelu3_w']['shape']
		self.data_prelu3w = np.array(data).reshape(shape_prelu3w)/mutil
		# conv4
		data = dict_h['conv4_w']['data']
		shape_conv4w = dict_h['conv4_w']['shape']
		self.data_conv4w = np.array(data).reshape(shape_conv4w)/mutil
		data = dict_h['conv4_b']['data']
		shape_conv4b = dict_h['conv4_b']['shape']
		self.data_conv4b = np.array(data).reshape(shape_conv4b)/mutil
		# prelu4
		data = dict_h['prelu4_w']['data']
		shape_prelu4w = dict_h['prelu4_w']['shape']
		self.data_prelu4w = np.array(data).reshape(shape_prelu4w)/mutil

		# conv5
		data = dict_h['conv5_w']['data']
		shape_conv5w = dict_h['conv5_w']['shape']
		self.data_conv5w = np.array(data).reshape(shape_conv5w)/mutil
		data = dict_h['conv5_b']['data']
		shape_conv5b = dict_h['conv5_b']['shape']
		self.data_conv5b = np.array(data).reshape(shape_conv5b)/mutil
		# prelu5
		data = dict_h['prelu5_w']['data']
		shape_prelu5w = dict_h['prelu5_w']['shape']
		self.data_prelu5w = np.array(data).reshape(shape_prelu5w)/mutil
		# conv11
		data = dict_h['conv11_w']['data']
		shape_conv11w = dict_h['conv11_w']['shape']
		self.data_conv11w = np.array(data).reshape(shape_conv11w)/mutil
		data = dict_h['conv11_b']['data']
		shape_conv11b = dict_h['conv11_b']['shape']
		self.data_conv11b = np.array(data).reshape(shape_conv11b)/mutil
	
	def forward(self, face):
		input_data = cv2.resize(face.copy(), (112,112))
		input_data = np.transpose(input_data, (2,0,1))
		input_data = np.expand_dims(input_data, axis=0)
		start = cv2.getTickCount()
		conv_out = Conv2D(input_data, input_data.shape,16,3,1,'VALID', self.data_conv1w, self.data_conv1b)
		conv_out = Prelu(conv_out, self.data_prelu1w)
		conv_out = pooling(conv_out, 3, 2, 1, 'max')

		conv_out = Conv2D(conv_out, conv_out.shape,32,3,1,'VALID', self.data_conv2w, self.data_conv2b)
		conv_out = Prelu(conv_out, self.data_prelu2w)
		conv_out = pooling(conv_out, 3, 2, 0, 'max')
		# conv3
		conv_out = Conv2D(conv_out, conv_out.shape,32,3,1,'VALID', self.data_conv3w, self.data_conv3b)
		conv_out = Prelu(conv_out, self.data_prelu3w)
		conv_out = pooling(conv_out, 2, 2, 0, 'max')
		# conv4
		conv_out = Conv2D(conv_out, conv_out.shape,64,2,1,'VALID', self.data_conv4w, self.data_conv4b)
		conv_out = Prelu(conv_out, self.data_prelu4w)
		conv_out = InnerProduct(conv_out, self.data_conv5w, self.data_conv5b)
		conv_out = Prelu(conv_out, self.data_prelu5w)
		# InnerProduct
		conv_out = InnerProduct(conv_out, self.data_conv11w, self.data_conv11b)
		softmax_out = Softmax(conv_out)
		end = cv2.getTickCount()
		return softmax_out


