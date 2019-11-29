import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from IPython.display import HTML
from base64 import b64encode
from math import floor

# video = cv2.VideoCapture('test_videos\solidWhiteRight.mp4')
# video = cv2.VideoCapture('test_videos\solidYellowLeft.mp4')
video = cv2.VideoCapture('test_videos\challenge.mp4')
nFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)
waitPerFrameInMillisec = int(1/fps * 1000/1)

kernel_size = 5
low_threshold = 120
high_threshold = 170

ignore_mask_color = 255
rho = 2
theta = np.pi/180
threshold = 60
min_line_length = 2
max_line_gap = 130

frames_list = []
# video_name = 'test_videos_output\output_solidWhiteRight.avi'
# video_name = 'test_videos_output\output_solidYellowLeft.avi'
video_name = 'test_videos_output\output_challenge.avi'
success, image = video.read()
height, width, layers = image.shape
size = (width, height)
out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

def improve_lane(last_line, shape, top_line):
    x1 = last_line[0]
    y1 = last_line[1]
    x2 = last_line[2]
    y2 = last_line[3]
    y_bottom, x, layer = shape
    y_top = top_line + 10

    m = ((y2 - y1)/(x2 - x1))

    x_top = floor((1/m) * (y_top - y1) + x1)
    x_bottom = floor((1/m) * (y_bottom - y1) + x1)

    return [x_top, y_top, x_bottom, y_bottom]

def run():
    success, image = video.read()
    if(success == True):
        line_image = np.copy(image) * 0
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
        imshape = image.shape

        mask = np.zeros_like(edges)
        top_line = 320      #320
        left_line = 460     #460
        right_line = 500    #500
        vertices = np.array([[(0, imshape[0]), (left_line, top_line), (right_line, top_line), (imshape[1], imshape[0])]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        max_value_r = 0
        min_value_r = 100000
        max_value_l = 0
        min_value_l = 100000
        last_line_l_top = None
        last_line_l_bottom = None
        last_line_r_top = None
        last_line_r_bottom = None
        left_line_list = []
        right_line_list = []
        for line in lines:
            left_l = (line[0][0] < left_line) * line
            right_l = (line[0][0] > right_line) * line
            print('right_l: ', right_l)
            if left_l[0][0] != 0:
                left_line_list.append(left_l)
                if max_value_l < left_l[0][1]:
                    max_value_l = left_l[0][1]
                    last_line_l_bottom = left_l[0]
                    if max_value_l < left_l[0][3]:
                        max_value_l = left_l[0][3]
                        last_line_l_bottom = left_l[0]

                if min_value_l > left_l[0][1]:
                    min_value_l = left_l[0][1]
                    last_line_l_top = left_l[0]
                    if min_value_l > left_l[0][3]:
                        min_value_l = left_l[0][3]
                        last_line_l_top = left_l[0]

            if right_l[0][0] != 0:
                right_line_list.append(right_l)
                if max_value_r < right_l[0][1]:
                    max_value_r = right_l[0][1]
                    last_line_r_bottom = right_l[0]
                    if max_value_r < right_l[0][3]:
                        max_value_r = right_l[0][3]
                        last_line_r_bottom = right_l[0]

                if min_value_r > right_l[0][1]:
                    min_value_r = right_l[0][1]
                    last_line_r_top = right_l[0]
                    if min_value_r > right_l[0][3]:
                        min_value_r = right_l[0][3]
                        last_line_r_top = right_l[0]

        # for line in lines:
        #     for x1, y1, x2, y2 in line:
        #         cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

        x1_t, y1_t, x2_t, y2_t = improve_lane(last_line_l_top, image.shape, top_line)
        x1_b, y1_b, x2_b, y2_b = improve_lane(last_line_l_bottom, image.shape, top_line)
        x1 = floor((x1_t + x1_b) * 0.5)
        y1 = floor((y1_t + y1_b) * 0.5)
        x2 = floor((x2_t + x2_b) * 0.5)
        y2 = floor((y2_t + y2_b) * 0.5)
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

        if last_line_l_top is None:
            print('last_line_l_top')

        if last_line_l_bottom is None:
            print('last_line_l_bottom')

        f_valid = True
        if last_line_r_top is None:
            f_valid = False

        if last_line_r_bottom is None:
            f_valid = False

        if f_valid == True:
            x1 = last_line_r_top[0]
            x2 = last_line_r_top[2]
            if (x2 - x1) != 0:
                x1 = last_line_r_bottom[0]
                x2 = last_line_r_bottom[2]
                if (x2 - x1) != 0:
                    x1_t, y1_t, x2_t, y2_t = improve_lane(last_line_r_top, image.shape, top_line)
                    x1_b, y1_b, x2_b, y2_b = improve_lane(last_line_r_bottom, image.shape, top_line)
                    x1 = floor((x1_t + x1_b) * 0.5)
                    y1 = floor((y1_t + y1_b) * 0.5)
                    x2 = floor((x2_t + x2_b) * 0.5)
                    y2 = floor((y2_t + y2_b) * 0.5)
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

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