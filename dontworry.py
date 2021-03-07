import cv2
import numpy as np
import random as rng
import imutils
import array as arr 
from pprint import pprint as pp

#need:
#if contour is smaller than a certain value, ie, in the background, then it isnt counted
#

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 15)
#cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
#cap.set(cv2.CAP_PROP_EXPOSURE, 50)
x1 = 3
x2 = 8
white = (255,255,255)

#test with orange
# lower_color = np.array([0, 180, 200], np.uint8)
# upper_color = np.array([30, 255, 255], np.uint8)

#test with the retroreflective, assumed blue
lower_color = np.array([148, 200, 200], np.uint8)
upper_color = np.array([180, 255, 255], np.uint8)

kernel = np.ones((x1,x1), np.uint8)
kernel2 = np.ones((x2,x2), np.uint8)

kernel = np.ones((1,8), np.uint8)


while True:
    frame = cap.read()[1]

    x_size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    y_size = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    black = np.zeros((y_size, x_size, 3), np.uint8)


    image = cv2.inRange(cv2.cvtColor(
        frame, cv2.COLOR_BGR2HSV), lower_color, upper_color)

    image = cv2.erode(image, kernel)
    image = cv2.dilate(image,kernel2)

    blur = cv2.GaussianBlur(image, (5, 5), 0)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

    # create hull array for convex hull points
    hull = []

    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))


    # create an empty black image
    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
    
    # draw contours and hull points
    for i in range(len(contours)):
        color_contours = (0, 255, 0) # green - color for contours
        color = (255, 0, 0) # blue - color for convex hull
        # draw ith contour
        cv2.drawContours(frame, contours, i, color_contours, 1, 8, hierarchy)
        # draw ith convex hull object
        cv2.drawContours(drawing, hull, i, color, 1, 8)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cXarr = []
    cYarr = []

    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cXarr.append(cX)
        cYarr.append(cY)
        # draw the contour and center of the shape on the image
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.circle(drawing, (cX, cY), 7, (255, 0, 255), -1)

        cv2.putText(drawing, str(cX), (cX - 20, cY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.putText(drawing, str(cY), (cX - 20, cY + 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 2)

    if len(cXarr)==2: #we have two contours
        #draw the line
        delta_x = abs(cXarr[0]-cXarr[1])
        delta_y = abs(cYarr[0]-cXarr[1])
        cv2.line(drawing,(cXarr[0],cYarr[0]),(cXarr[1],cYarr[1]),(255,0,0),1)
    else:
        delta_x = 0
        delta_y = 0

    cv2.putText(drawing, str(delta_x)+ " - " + str(delta_y), (10,15), 
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 2)


    cv2.imshow("raw_image", frame)
    cv2.imshow("mask1", image)
    cv2.imshow("drawing", drawing)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()