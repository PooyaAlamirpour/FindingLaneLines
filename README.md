# SELF-DRIVING CAR: Finding Lane Lines
In this repository, I am going to implement Lane Lines Detection on a video. At the beginning of implementing this project, we have to set up some requirements. I recommend you see [this online](https://classroom.udacity.com/courses/ud1111 "Anaconda and Jupyter notebooks") free course in the Udacity for more information.
Of course, I am using [Google Colab](https://colab.research.google.com/). It is easy and free.

[Anaconda](https://anaconda.org/) is a distribution of packages built for data science. It comes with conda, a package and environment manager. You'll be using conda to create environments for isolating your projects that use different versions of Python and/or different packages. You'll also use it to install, uninstall, and update packages in your environments. Using Anaconda has made my life working with data much more pleasant.


### STEPS TO FIND LANE LINES
Useful features for identifying lane lines: color, shape, orientation, position. We’ll start with color detection, then region masking, then finding edges, and finally using a Hough Transform to identify line segments.

### COLOR SELECTION
Values from 0 (dark) to 255 (bright) in Red, Green, and Blue color channels.

![Color Selection](https://github.com/PooyaAlamirpour/FindingLaneLines/blob/master/Pictures/COLOR_SELECTION.png)

### REGION MASKING
Add criteria in code to only focus on a specific region of an image, since lane lines will always appear in the same general part of the image.

After running `ColorLaneLinesDetection.py`, you must see the below picture as a result. This code extracts white pixels as lane lines. Then by putting a triangular mask, we can limit the area of exploring.
```python
plt.imshow(region_select)
```
![Color Selection](https://github.com/PooyaAlamirpour/FindingLaneLines/blob/master/Pictures/mask_color.png)

```python
plt.imshow(line_image)
```
![Color Selection](https://github.com/PooyaAlamirpour/FindingLaneLines/blob/master/Pictures/final_color_mask.png)

### CANNY EDGE DETECTION
Find edges by looking for strong gradient, i.e. very different values between adjacent pixels.
```python
edges = cv2.Canny(gray, low_threshold, high_threshold)
```

### HOUGH TRANSFORM
A line in image space can be represented as a single point in parameter space, or Hough Space.

![Hough Transform](https://github.com/PooyaAlamirpour/FindingLaneLines/blob/master/Pictures/HOUGH_TRANSFORM.png)

### Putting togather


### REFERENCES
* [A Gentle Introduction to Computer Vision](https://machinelearningmastery.com/what-is-computer-vision/)
* [Udacity free course ​Intro to Computer Vision](https://www.udacity.com/course/introduction-to-computer-vision--ud810)
* [OpenCV](https://opencv.org/)