# SELF-DRIVING CAR: Finding Lane Lines
In this repository, I am going to implement Lane Lines Detection on a video. At the beginning of implementing this project, we have to set up some requirements. I recommend you see [this online](https://classroom.udacity.com/courses/ud1111 "Anaconda and Jupyter notebooks") free course in the Udacity for more information about setting up suitable environment.
Of course, I am using [Google Colab](https://colab.research.google.com/). It is easy and free.

There is another powerful tool is named [Anaconda](https://anaconda.org/). Anaconda is a distribution of packages built for data science. It comes with conda, a package and environment manager. You'll be using conda to create environments for isolating your projects that use different versions of Python and/or different packages. You'll also use it to install, uninstall, and update packages in your environments. Using Anaconda has made my life working with data much more pleasant.
So it depends on you which tool will be selected.

### STEPS TO FIND LANE LINES
For lane detection on a video, we have to pass some steps that you can see below:
* Getting each frame from video
* Making grayscale each frame
* Detecting edges by using Canny Algorithm
* Finding Lane by using Hough Algorithm
* Improving output and making a new video as a result

At the beginning of this project let me show you why we can not use just color segmentation for finding a lane in a video. As clear, often the color of the lane is white. When daylight is enough you can detect white lanes easily, But think about the night. It is hardly possible to detect the lanes. But it is not the only reason to refuse the color segmentation algorithm. You can imagine there are lots of objects would be either near white or completely white in the real world. Objects have some features such as color, shape, orientation, position, etc that would be helpful for us for detecting them.
Anyway, I want to show you practically by using python and OpenCV which is a powerful tool for image processing.

### COLOR SELECTION
Values from 0 (dark) to 255 (bright) in Red, Green, and Blue color channels. it is possible we extract lane lines by choosing just white pixels. But there are some areas that have white pixels that do not belong to the lane line. For eliminating these areas we can use region masking. Add criteria in code to only focus on a specific region of an image, since lane lines will always appear in the same general part of the image. So it is another problem of Color Segmentation Algorithm. 
If you want to test practically just clone this repository and use `ColorLaneLinesDetection.py`. After running `ColorLaneLinesDetection.py`, you must see the below pictures as the result. This code extracts white pixels as lane lines. Then by putting a triangular mask, we can limit the area of exploring.
```python
plt.imshow(region_select)
```
![REGION MASKING](https://github.com/PooyaAlamirpour/FindingLaneLines/blob/master/Pictures/mask_color.png)

```python
plt.imshow(line_image)
```
![REGION MASKING](https://github.com/PooyaAlamirpour/FindingLaneLines/blob/master/Pictures/final_color_mask.png)

So let's continue our steps to find lane lines. We'll start with the Canny Algorithm. 

### CANNY EDGE DETECTION
The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in images. It was developed by John F. Canny in 1986. Canny also produced a computational theory of edge detection explaining why the technique works.
An edge in an image may point in a variety of directions, so the Canny algorithm uses four filters to detect horizontal, vertical and diagonal edges in the blurred image by Finding the intensity gradients of the image.
After applying gradient calculation, the edge extracted from the gradient value is still quite blurred. Thus non-maximum suppression can help to suppress all the gradient values (by setting them to 0) except the local maxima, which indicate locations with the sharpest change of intensity value.

```python
edges = cv2.Canny(gray, low_threshold, high_threshold)
```

For exploring of Canny Edge Detection, run `CannytoDetectLaneLines.py` code. After successfully running you must see the below picture as the result.
```python
plt.imshow(edges, cmap='Greys_r')
```
![CANNY EDGE DETECTION](https://github.com/PooyaAlamirpour/FindingLaneLines/blob/master/Pictures/canny_edges.png)

As you can see, just the important lines of the image which have strong edges have remained. The main question is, how we can extract straight lines from an image. The Hough Algorithm can answer this question.

### HOUGH TRANSFORM
The Hough transform is a feature extraction technique used in image analysis, computer vision, and digital image processing. The purpose of the technique is to find imperfect instances of objects within a certain class of shapes by a voting procedure. This voting procedure is carried out in a parameter space, from which object candidates are obtained as local maxima in a so-called accumulator space that is explicitly constructed by the algorithm for computing the Hough transform.
The classical Hough transform was concerned with the identification of lines in the image, but later the Hough transform has been extended to identifying positions of arbitrary shapes, most commonly circles or ellipses.
A line in image space can be represented as a single point in parameter space, or Hough Space. We use this theory for detecting lines in a picture. So for achieving this goal, we should add the result of the Canny Algorithm to Hough.

![Hough Transform](https://github.com/PooyaAlamirpour/FindingLaneLines/blob/master/Pictures/HOUGH_TRANSFORM.png)

Hough Algorithm has some parameters which have role key in tuning algorithm fine. You can either put a long time for tuning parameters of algorithm or put an especial mask for eliminating other unuseful areas from the picture. Check the difference between using masked area and just tuning parameters of the Hough Algorithm. In the below images you can see the detected lines as red color.

Without Area Selection(unMasked)
```python
vertices = np.array([[(0, imshape[0]), (0, 0), (imshape[1], 0), (imshape[1], imshape[0])]], dtype=np.int32)
```
![HOUGH TRANSFORM](https://github.com/PooyaAlamirpour/FindingLaneLines/blob/master/Pictures/hough_without_mask.png)

Suitable Area Selection(Masked)
```python
vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
```
![HOUGH TRANSFORM](https://github.com/PooyaAlamirpour/FindingLaneLines/blob/master/Pictures/hough_with_mask.png)


### Putting togather
Now it is time to put all our knowledge together for finding lanes in a video. This video is a source for testing our algorithm for finding lanes. You can find `solidWhiteRight.mp4` inside of the `test_videos` folder.
Click on the below image and see the source video.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=Bp-uvoz74hs" target="_blank"><img src="https://github.com/PooyaAlamirpour/FindingLaneLines/blob/master/Pictures/raw_solidWhiteRight.png" alt="solidWhiteRight.mp4" width="375" height="223" border="10" /></a>

Processing a video is as same as processing an image; the difference is that the video consists of lots of sequential images. So at first, we must load and extract all frames of a video.

```python
video = cv2.VideoCapture('test_videos\solidWhiteRight.mp4')
# This line extracts just one frame in each run.
frame = video.read()
```

In this project, it is used, Canny edge detection and hough algorithm. Both of them have some parameters that it has to tune first. I use below value for their parameters.

```python
kernel_size = 5
low_threshold = 120
high_threshold = 170

ignore_mask_color = 255
rho = 2
theta = np.pi/180
threshold = 50
min_line_length = 2
max_line_gap = 130
```

If you want to run this algorithm clone this repository and run `main.py`. You can see the result as below. The blue lines show the detected lane on the road. Click on the image for watching the video result.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=c5Q6f3Jgbqw" target="_blank"><img src="https://github.com/PooyaAlamirpour/FindingLaneLines/blob/master/Pictures/result_Solid_white_right.png" alt="output_solidWhiteRight.avi" width="375" height="223" border="10" /></a>

If you watch the above video you will see there is a gap between the bottom of the video and the last lane. 

![Improving](https://github.com/PooyaAlamirpour/FindingLaneLines/blob/master/Pictures/gap_SolidWhiteRight.png)

Because the hough algorithm is not able to find a destination point at the bottom of the video. So the line would not be continuous. For solving this issue, I recommend a simple solution.  As you know, the output of the Hough Algorithm consists of some lines. I found the last left line and try to continue that line by using the line equation.

```python
for line in lines:
	left_l = (line[0][0] < left_line) * line
	if left_l[0][0] != 0:
		left_line_list.append(left_l)
	if max_value < left_l[0][1]:
		max_value = left_l[0][1]
		last_line = left_l[0]
		if max_value < left_l[0][3]:
			max_value = left_l[0][3]
			last_line = left_l[0]
```

The line equation was implemented in a function is named improve_lane.

```python
x1, y1, x2, y2 = improve_lane(last_line, image.shape)
cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 17)
```

```python
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
```
You can see the final result in the below video.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=7Kki_pW9k7A" target="_blank"><img src="https://github.com/PooyaAlamirpour/FindingLaneLines/blob/master/Pictures/result_Solid_white_right.png" alt="output_solidWhiteRight.avi" width="375" height="223" border="10" /></a>

Now, we have a suitable idea for improving our solution. As you know, the Hough Algorithm has multiple outputs. It means for a line in an image there would be lots of output from the Hough Algorithm. We can either calculate the average between all of the outputs or using the line equation for modelling and drawing just one line. You can use both of them. In this project, I used the second solution. For running this idea practically, you should clone this repository and run the `main_2.py`. You can see video result below:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=bw_wWoEjOqQ" target="_blank"><img src="https://github.com/PooyaAlamirpour/FindingLaneLines/blob/master/Pictures/result_Solid_white_right.png" alt="output_solidWhiteRight.avi" width="375" height="223" border="10" /></a>

for implementing that goal I changed the `improve_lane` function.

```python
def improve_lane(last_line, shape, top_line):
    x1 = last_line[0]
    y1 = last_line[1]
    x2 = last_line[2]
    y2 = last_line[3]
    y_bottom, x, layer = shape
    y_top = top_line + 10

    m = ((y2 - y1)/(x2 - x1))

    x_top = floor( (1/m) * (y_top - y1) + x1)
    x_bottom = floor((1/m) * (y_bottom - y1) + x1)

    return [x_top, y_top, x_bottom, y_bottom]
```

You can see the using of the new `improve_lane` function below:

```python
x1_t, y1_t, x2_t, y2_t = improve_lane(last_line_l_top, image.shape, top_line)
x1_b, y1_b, x2_b, y2_b = improve_lane(last_line_l_bottom, image.shape, top_line)
x1 = floor((x1_t + x1_b) * 0.5)
y1 = floor((y1_t + y1_b) * 0.5)
x2 = floor((x2_t + x2_b) * 0.5)
y2 = floor((y2_t + y2_b) * 0.5)
cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

x1_t, y1_t, x2_t, y2_t = improve_lane(last_line_r_top, image.shape, top_line)
x1_b, y1_b, x2_b, y2_b = improve_lane(last_line_r_bottom, image.shape, top_line)
x1 = floor((x1_t + x1_b) * 0.5)
y1 = floor((y1_t + y1_b) * 0.5)
x2 = floor((x2_t + x2_b) * 0.5)
y2 = floor((y2_t + y2_b) * 0.5)
cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
```

By using this idea we can even detect the yellow lane in a video continuously. So it does not matter the dash-line is the left side or the right side of the screen. We can extract the lane from both sides of the screen.

### REFERENCES
* [A Gentle Introduction to Computer Vision](https://machinelearningmastery.com/what-is-computer-vision/)
* [Udacity free course ​Intro to Computer Vision](https://www.udacity.com/course/introduction-to-computer-vision--ud810)
* [OpenCV](https://opencv.org/)
* [Canny edge detector](https://en.wikipedia.org/wiki/Canny_edge_detector)
* [Hough transform](https://en.wikipedia.org/wiki/Hough_transform)