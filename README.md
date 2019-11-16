# SELF-DRIVING CAR: Finding Lane Lines

## STEPS TO FIND LANE LINES
Useful features for identifying lane lines: color, shape, orientation, position. We’ll start with color detection, then region masking, then finding edges, and finally using a Hough Transform to identify line segments.

## COLOR SELECTION
Values from 0 (dark) to 255 (bright) in Red, Green, and Blue color channels.
![alt text](http://url/to/img.png)

## REGION MASKING
Add criteria in code to only focus on a specific region of an image, since lane lines will always appear in the same general part of the image.

## CANNY EDGE DETECTION
Find edges by looking for strong gradient, i.e. very different values between adjacent pixels.
```python
edges = cv2.Canny(gray, low_threshold, high_threshold)
```

## HOUGH TRANSFORM
A line in image space can be represented as a single point in parameter space, or Hough Space.
![alt text](http://url/to/img.png)