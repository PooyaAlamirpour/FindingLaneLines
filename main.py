import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from IPython.display import HTML
from base64 import b64encode
from math import floor

video = cv2.VideoCapture('test_videos\solidWhiteRight.mp4')
nFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)
waitPerFrameInMillisec = int(1/fps * 1000/1)

kernel_size = 5
low_threshold = 120
high_threshold = 170

ignore_mask_color = 255
rho = 2
theta = np.pi/180
threshold = 50
min_line_length = 2
max_line_gap = 130     

frames_list = []
video_name = 'test_videos_output\output_solidWhiteRight.avi'
success, image = video.read()
height, width, layers = image.shape
size = (width, height)
out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

def improve_lane(last_line, shape):
    x1 = last_line[0]
    y1 = last_line[1]
    x2 = last_line[2]
    y2 = last_line[3]
    y, x, layer = shape
    m = ((y2 - y1)/(x2 - x1))
    x = (1/m)*(y - y1) + x1
    x = floor(x)
    return [x1, y1, x, y]

def run():
    success, image = video.read()
    if(success == True):
        line_image = np.copy(image) * 0
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
        imshape = image.shape

        mask = np.zeros_like(edges)
        top_line = 320
        left_line = 460
        right_line = 500
        vertices = np.array([[(0, imshape[0]), (left_line, top_line), (right_line, top_line), (imshape[1], imshape[0])]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        max_value = 0
        last_line = None
        left_line_list = []
        for line in lines:
            # print('line: ', line, '-', (line[0][0] < left_line) * line)
            left_l = (line[0][0] < left_line) * line
            if left_l[0][0] != 0:
                left_line_list.append(left_l)
            if max_value < left_l[0][1]:
                max_value = left_l[0][1]
                last_line = left_l[0]
                if max_value < left_l[0][3]:
                    max_value = left_l[0][3]
                    last_line = left_l[0]

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

        x1, y1, x2, y2 = improve_lane(last_line, image.shape)
        # print('x1: ', x1, '-y1:', y1, '-x2:', x2, '-y2:', y2)
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 17)

        color_edges = np.dstack((edges, edges, edges))
        lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
        final_image = cv2.addWeighted(line_image, 0.8, image, 1, 0)
        out.write(final_image)

        image = None
        line_image = None

# run()
for i in range(0, nFrames - 1):
    run()


out.release()

print('Finished: ', len(frames_list))