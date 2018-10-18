# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:05:50 2018

@author: wmy
"""

import cv2

cameraCapture = cv2.VideoCapture(0)
fps = 30
size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter('MyOutputVid.avi',
                              cv2.VideoWriter_fourcc('I', '4', '2', '0'),
                              fps,
                              size)

success, frame = cameraCapture.read()
numFramesRemaining = 10 * fps - 1
while success and numFramesRemaining > 0:
    videoWriter.write(frame)
    success, frame = cameraCapture.read()
    numFramesRemaining -= 1
    pass
cameraCapture.release()
