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
A line in image space can be represented as a single point in parameter space, or Hough Space. We use this theory for detecting lines in a picture. So for achieving this goal, we should add the result of the Canny Algorithm to Hough.

![Hough Transform](https://github.com/PooyaAlamirpour/FindingLaneLines/blob/master/Pictures/HOUGH_TRANSFORM.png)

Hough Algorithm has some parameters which have role key in tuning algorithm fine. You can either put a long time for tuning parameters of algorithm or put an especial mask for eliminating other unuseful areas from the picture. Check the difference between using masked area and just tuning parameters of the Hough Algorithm.

Big area selection(unMasked)
```python
vertices = np.array([[(0, imshape[0]), (0, 0), (imshape[1], 0), (imshape[1], imshape[0])]], dtype=np.int32)
```
![HOUGH TRANSFORM](https://github.com/PooyaAlamirpour/FindingLaneLines/blob/master/Pictures/hough_without_mask.png)

Suitable area selection(Masked)
```python
vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
```
![HOUGH TRANSFORM](https://github.com/PooyaAlamirpour/FindingLaneLines/blob/master/Pictures/hough_with_mask.png)

### Putting togather


### REFERENCES
* [A Gentle Introduction to Computer Vision](https://machinelearningmastery.com/what-is-computer-vision/)
* [Udacity free course ​Intro to Computer Vision](https://www.udacity.com/course/introduction-to-computer-vision--ud810)
* [OpenCV](https://opencv.org/)
* [Canny edge detector](https://en.wikipedia.org/wiki/Canny_edge_detector)