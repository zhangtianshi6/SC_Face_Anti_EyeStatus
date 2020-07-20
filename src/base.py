# -*- coding: utf-8 -*-
# @Time : 20-7-20
# @Author : zhangT
# @Company : *
# @File : base.py
import numpy as np
import math
import cv2


def im2col(image, k_size, stride):
    image_col = []
    for i in range(0, image.shape[1] - k_size+1, stride):
        for j in range(0, image.shape[2]-k_size+1, stride):
            col = image[:, i:i+k_size, j:j+k_size].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)
    return image_col


#####################################
# convolution                       #
#####################################
def Conv2D(x, shape, output_channels, ksize=3, stride=1, method='VALID', weights=None, bias=None):
    batchsize = shape[0]
    col_weights = weights.reshape([output_channels, -1])
    if method == 'SAME':
        x = np.pad(x, ((0, 0), (0, 0), (ksize // 2, ksize // 2),
                       (ksize // 2, ksize // 2)), 'constant', constant_values=0)
        eta = np.zeros(
            (shape[0], output_channels, shape[2]//stride, shape[3]//stride))
    if method == 'VALID':
        eta = np.zeros((shape[0], output_channels, (
            shape[2] - ksize) // stride + 1, (shape[3] - ksize) // stride + 1))

    col_image = []
    conv_out = np.zeros(eta.shape)
    for i in range(batchsize):
        img_i = x[i]
        col_image_i = im2col(img_i, ksize, stride)
        col_image_i = np.transpose(col_image_i, (1, 0))
        temp = np.dot(col_weights, col_image_i)
        conv_out[i] = np.array([np.reshape(
            temp[a]+bias[a], (eta[0].shape[1], eta[0].shape[2])) for a in range(temp.shape[0])])
        col_image.append(col_image_i)
    return conv_out


#####################################
# Prelu                             #
# input_param                       #
# -inputs: input data               #
# -preluw: relu w                   #
# output_param                      #
# -outputs: relu output             #
#####################################
def Prelu(inputs, preluw):
    shape = inputs.shape
    if inputs.ndim == 4:
        for i in range(shape[0]):
            for j in range(shape[1]):
                inputs[i, j] = np.where(
                    inputs[i, j] < 0, preluw[j]*inputs[i, j], inputs[i, j])
    elif inputs.ndim == 2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                data = inputs[i, j]
                inputs[i, j] = np.where(
                    inputs[i, j] < 0, preluw[j]*inputs[i, j], inputs[i, j])
    else:
        print("Prelu fail")
    return inputs


#####################################
# relu                              #
# input_param                       #
# -inputs: input data               #
# output_param                      #
# -outputs: relu output             #
#####################################
def Relu(inputs):
    shape = inputs.shape
    if inputs.ndim == 4:
        for i in range(shape[0]):
            for j in range(shape[1]):
                inputs[i, j] = np.where(inputs[i, j] < 0, 0, inputs[i, j])
    elif inputs.ndim == 2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                data = inputs[i, j]
                inputs[i, j] = np.where(inputs[i, j] < 0, 0, inputs[i, j])
    else:
        print("Prelu fail")
    return inputs


#####################################
# pooling                           #
# input_param                       #
# -inputs: input data               #
# -ksize: pooling size              #
# -pad: pooling pad                 #
# -method: pooling method           #
# output_param                      #
# -outputs: pooling output          #
#####################################
def pooling(inputs, ksize=3, stride=2, pad=0, method='max'):
    outputs = None
    strides = stride
    pads = pad
    pool_H = ksize
    pool_W = ksize
    (N, C, H, W) = inputs.shape
    inputs_reshaped = inputs.reshape(N * C, 1, H, W)
    (N_P, N_C, N_H, N_W) = inputs_reshaped.shape
    # calculating the height and width of next layer using formula
    if pad:
        inputs_reshaped_pad = np.pad(
            inputs_reshaped, ((0, 0), (0, 0), (0, 2*pads), (0, 2*pads)), 'edge')
    else:
        inputs_reshaped_pad = inputs_reshaped

    out_height = 1 + int((H + 2 * pads - pool_H) / strides)
    out_width = 1 + int((W + 2 * pads - pool_W) / strides)

    i0 = np.repeat(np.arange(pool_H), pool_W)
    i0 = np.tile(i0, N_C)
    i1 = strides * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(pool_W), pool_H * N_C)
    j1 = strides * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(N_C), pool_H * pool_W).reshape(-1, 1)

    cols = inputs_reshaped_pad[:, k, i, j]
    X_col = cols.transpose(1, 2, 0).reshape(pool_H * pool_W * N_C, -1)

    if method == 'max':
        max_ind = np.argmax(X_col, axis=0)
        outputs = X_col[max_ind, range(max_ind.size)]
    elif method == 'avg':
        avg = np.mean(X_col, axis=0)
        outputs = avg

    outputs = outputs.reshape(out_height, out_width, N, C)
    outputs = outputs.transpose(2, 3, 0, 1)

    return outputs


#####################################
# connection layer                  #
# input_param                       #
# -innerw: connection's w           #
# -inner_bias: connection's b       #
# output_param                      #
# -inner_out: connection's output   #
#####################################
def InnerProduct(inputs, innerw, inner_bias):
    batchsize = inputs.shape[0]
    inner_out = np.zeros((batchsize, innerw.shape[0],))
    for i in range(batchsize):
        x_temp = inputs[i]
        x_temp = x_temp.reshape([-1])
        if x_temp.shape[0] != innerw.shape[1]:
            return None
        else:
            for k in range(innerw.shape[0]):
                inner_out[i, k] = np.dot(x_temp, innerw[k])+inner_bias[k]
    return inner_out


#####################################
# softmax layer                     #
# input_param                       #
# -x: connection's w                #
# -inner_bias: connection's b       #
# output_param                      #
# -inner_out: connection's output   #
#####################################
def Softmax(inputs):
    softmax_out = np.zeros((inputs.shape[0], inputs.shape[1]))
    if inputs.ndim == 2:
        for i in range(inputs.shape[0]):
            inputs -= np.max(inputs)
            softmax_out[i] = np.exp(inputs) / np.sum(np.exp(inputs))
        return softmax_out
    else:
        return None
