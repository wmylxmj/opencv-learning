# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 18:59:52 2018

@author: wmy
"""

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

image = cv2.imread("16.jpg")
imagePlt = plt.imread('16.jpg')

'''
cv2.namedWindow('Image')
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.distroyAllWindows()
'''

R = cv2.split(image)[0]
G = cv2.split(image)[1]
B = cv2.split(image)[2]

'''
cv2.imshow("Image", image)
cv2.imshow("Red", R)
cv2.imshow("Green", G)
cv2.imshow("Blue", B)
cv2.waitKey(0)
'''

def calcAndDrawHist(image, color):
    hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImage = np.zeros([256, 256, 3], np.uint8)
    hpt = int(0.9*256)
    for h in range(256):
        intensity = int(hist[h]*hpt/maxVal)
        cv2.line(histImage, (h, 256), (h, 256-intensity), color)
        pass
    return histImage

histImageR = calcAndDrawHist(R, [255, 0, 0])
histImageG = calcAndDrawHist(G, [0, 255, 0])
histImageB = calcAndDrawHist(B, [0, 0, 255])

'''
plt.imshow(histImageR)
plt.show()
plt.imshow(histImageG)
plt.show()
plt.imshow(histImageB)
plt.show()
'''

'''
cv2.imshow('histImageR', histImageR)
cv2.imshow('histImageG', histImageG)
cv2.imshow('histImageB', histImageB)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.distroyAllWindows()
'''
 
#构造一个结构元素 
kernel = np.ones((3, 3), np.uint8)

dilate = cv2.dilate(image, kernel, iterations = 1)
erode = cv2.erode(image, kernel, iterations = 1)
 
#将两幅图像相减获得边，第一个参数是膨胀后的图像，第二个参数是腐蚀后的图像
result = cv2.absdiff(dilate,erode);
'''
#上面得到的结果是灰度图，将其二值化以便更清楚的观察结果
retval, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY); 
'''

#反色，即对二值图每个像素取反
result = cv2.bitwise_not(result); 

plt.imshow(imagePlt)
plt.show()

plt.imshow(result)
plt.show()

'''
#显示图像
cv2.imshow("result",result); 
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

full_shape = []
full_shape.append(1)
full_shape.append(imagePlt.shape[0])
full_shape.append(imagePlt.shape[1])
full_shape.append(imagePlt.shape[2])

full = np.reshape(imagePlt, full_shape)

filterArray = [[ 0.1,  0.1,  0.1],
               [-1.0, -1.0, -1.0],
               [-1.0, -1.0, -1.0],
               [ 1.0,  1.0,  1.0],
               [ 0.1,  0.1,  0.1],
               [-1.0, -1.0, -1.0],
               [ 1.0,  1.0,  1.0],
               [ 1.0,  1.0,  1.0],
               [ 0.1,  0.1,  0.1]]

myFilter = tf.Variable(tf.constant(filterArray, shape = [3, 3, 3, 1]))
inputfull = tf.Variable(tf.constant(1.0, shape = full_shape))
op = tf.nn.conv2d(inputfull, myFilter, strides=[1, 1, 1, 1], padding='SAME')
o = tf.cast(((op-tf.reduce_min(op))/(tf.reduce_max(op)-tf.reduce_min(op)))*255, tf.uint8)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    t, f = sess.run([o, myFilter], feed_dict={inputfull : full})
    print(t.shape)
    t = np.reshape(t, [imagePlt.shape[0], imagePlt.shape[1]])
    imagevector = np.array(t)
    height,width = imagevector.shape
    for h in range(height):
        for w in range(width):
            if imagevector[h,w]<=128:
                imagevector[h,w]=0
            else:
                imagevector[h,w]=1
                pass
            pass
        pass
    plt.imshow(imagevector, cmap='gray')
    plt.axis('on')
    plt.show()
    pass

imageFilted = imagevector
reluStep = tf.nn.relu(imageFilted)

with tf.Session() as sess:
    sess.run(init)
    reluResult = sess.run(reluStep)
    plt.imshow(reluResult, cmap='gray')
    plt.axis('on')
    plt.show()
    
reluResult = np.reshape(reluResult, (1, reluResult.shape[0], reluResult.shape[1], 1))
print(reluResult.shape)

maxpoolStep = tf.nn.max_pool(reluResult, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.Session() as sess:
    sess.run(init)
    maxpoolResult = sess.run(maxpoolStep)
    print(maxpoolResult.shape)
    maxpoolResult = np.reshape(maxpoolResult, [maxpoolResult.shape[1], maxpoolResult.shape[2]])
    plt.imshow(maxpoolResult, cmap='gray')
    plt.axis('on')
    plt.show()    

    
    