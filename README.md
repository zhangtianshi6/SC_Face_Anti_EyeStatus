# Name: SC_Face_Anti_EyeStatus

It's for Face Anti, Eye Status and landmark model for RGB img, run in windows 32.
All code only rely on numpy and opencv to run. 

# install
numpy
opencv


# before:
if you want to convert caffemodel to param.py
change caffe path, model name, conv name ……
run python caffe_convert/caffe_to_param.py

# run:
test.py

# Anti
Model        ACC   GFlops     Weights 
MiniFAntiV1  0.981  3.2     4.1M

