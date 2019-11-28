import cv2
import numpy as np
import matplotlib.pyplot as plt

video = cv2.VideoCapture('test_videos\solidWhiteRight.mp4')
frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

pathOut = 'video.avi'
fps = 25
success, image = video.read()
height, width, layers = image.shape
size = (width, height)

out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(0, 20):
    success, image = video.read()
    out.write(image)
    print('type: ', type(image))
out.release()