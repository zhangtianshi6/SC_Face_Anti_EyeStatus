# -*- coding: utf-8 -*-
# @Time : 20-7-20
# @Author : zhangT
# @Company : *
# @File : caffe_to_param.py
import caffe
import sys
import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
caffe_root = '/path/to/caffe/'
sys.path.insert(0, caffe_root + 'python')
deploy = 'model/*.prototxt'
caffe_model = 'model/*.caffemodel'
net = caffe.Net(deploy, caffe_model, caffe.TEST)
params = [(k, v[0].data) for k, v in net.params.items()]

for i, v in params:
    for k in net.params[i]:
        print(i, k.shape)

fw = open('params.py', 'w')
fw.write('dict_h={')
mutil = 100000000
for i, v in params:
    w1 = np.array(net.params[i][0].data, dtype=np.float32)
    print(i, str(w1.shape))
    shape = w1.shape
    length = 1
    for k in range(w1.ndim):
        length *= shape[k]
    info = '\''+i+'_w\':{ \'shape\':'+str(w1.shape)+', \'data\':['

    for p in w1:
        if 'conv' in i:

            for k in range(w1.shape[1]):
                for m in range(w1.shape[2]):
                    for n in range(w1.shape[3]):
                        #print(p[k][m][n], str(p[k][m][n]))
                        info += str(p[k][m][n]*mutil)+', '
        elif 'relu' in i:
            new_p = str(p*mutil).replace('\n', ' ')
            info += str(new_p+', ')
            #print(p, i)
        else:
            for k in range(w1.shape[1]):
                info += str(p[k]*mutil)+', '

    fw.write(info+']}, ')

    if 'conv' in i or 'inner' in i:
        b1 = np.array(net.params[i][1].data, dtype=np.float32)
        info = '\''+i+'_b\':{ \'shape\':'+str(b1.shape)+', \'data\':['
        for p in b1:
            new_p = str(p*mutil).replace('\n', ' ')
            info += str(new_p+', ')
        fw.write(info+']}, ')
fw.write('}\n')
print('finish')
