import numpy as np
import cv2


def judge_angle(all_points, list_max, box):

    nose_points = all_points[30]
    nose_28 = all_points[27]
    mouth_58 = all_points[57]
    nose_x, nose_y = nose_points
    all_points = np.array([all_points[a,:] for a in range(all_points.shape[0]) if list_max[a] > 0 ])
    x_max = np.max(all_points[:,0])
    x_min = np.min(all_points[:,0])
    y_max = np.max(all_points[:,1])
    y_min = np.min(all_points[:,1])
    left_right = 0
    up_down = 0
    rotate = 0
    if nose_x == x_max:
        left_right = 90
    if nose_x == x_min:
        left_right = -90
    if nose_y == y_max:
        up_down = -90
    if nose_y == y_min:
        up_down = 90
    # left and right
    dist_right = x_max-nose_x
    dist_left = nose_x-x_min
    x_len = x_max-x_min
    if dist_right>0 and dist_left>0:
        if dist_right > dist_left:
            left_right = -np.arccos(1-np.power((1-dist_left/x_len*2),2))*180/np.pi
        else:
            left_right = np.arccos(1-np.power((1-dist_right/x_len*2),2))*180/np.pi
    
    # up and down
    dist_down = box[3]-nose_y
    dist_up = nose_y-box[1]
    y_len = box[3]-box[1]
    if dist_up>0 and dist_down>0:
        if dist_up/108 > dist_down/62:
            rate = dist_down/y_len*3.3907127504730914-0.2010227897931709
            rate = 1 if rate > 1 else rate
            rate = 0 if rate < 0 else rate
            up_down = -np.arccos(1-np.power((1-rate),2))*180/np.pi
        else:
            rate = dist_up/y_len*2.050007695212636-0.32387474119124426
            rate = 1 if rate > 1 else rate
            rate = 0 if rate < 0 else rate
            up_down = np.arccos(1-np.power((1-rate),2))*180/np.pi
    
    # rotate
    left_eye, right_eye = eye_place(all_points)
    rotate = 0
    if right_eye[0]!=left_eye[0]:
        rotate = np.arctan((right_eye[1]-left_eye[1])/(right_eye[0]-left_eye[0]))*180/np.pi
    return left_right, up_down, rotate

def blurness(face):
    face = cv2.resize(face, (30, 30))
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_8U).var()
    return np.sqrt(score)

def illumination(image):
    image = cv2.resize(image, (30, 30))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_part = cv2.inRange(gray, 0, 30)
    bright_part = cv2.inRange(gray, 220, 255)
    #print(gray.shape, bright_part.shape, dark_part.shape)
    total_pixel = np.size(gray)
    dark_pixel = np.sum(dark_part > 0)
    bright_pixel = np.sum(bright_part > 0)
    dark_score = float(dark_pixel)/total_pixel
    bright_score = float(bright_pixel)/total_pixel
    score = (bright_score-dark_score+1)/2.0*255 
    return score



def box_adge(box, img_shape):
    # print(box)
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    dist_x1 = x1*1.0/img_shape[1]
    dist_y1 = y1*1.0/img_shape[0]
    dist_x2 = (img_shape[1]-x2)*1.0/img_shape[1]
    dist_y2 = (img_shape[0]-y2)*1.0/img_shape[0]
    thr = 0.005
    if dist_x1<thr or dist_y1<thr or dist_x2<thr or dist_y2<thr:
        return True
    return False




