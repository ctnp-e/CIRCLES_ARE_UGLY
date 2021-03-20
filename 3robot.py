import cv2
import numpy as np
import random as rng
import imutils
import array as arr 
from pprint import pprint as pp
import math
import threading
from networktables import NetworkTables

#I HAVE NOW JUST REALIZED WE ARE ONLY TURNING RIGHT THE WHOLE DAMN TIME

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 15)
#cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
#cap.set(cv2.CAP_PROP_EXPOSURE, 50)
x1 = 3
x2 = 8
white = (255,255,255)
#minimum area that is accepted.
min_area = 2000

#movement responses:
#1 = move foward
#2 = turn right
#3 = 180
movement = 0
value = -1

def do(x):
    what_it_do = {
        1: "move foward",
        2: "180",
        3: "turn"
    }
    print (what_it_do.get(x, " "))

#stole this part from henry. i hope it connects
#https://github.com/2643/2020-vision/blob/multi-target-dev/main.py 53-76
def connect():
    cond = threading.Condition()
    notified = [False]

    def connectionListener(connected, info):
        print(info, '; Connected=%s' % connected)
        with cond:
            notified[0] = True
            cond.notify()

    NetworkTables.initialize(server='roborio-2643-frc.local')
    NetworkTables.addConnectionListener(
        connectionListener, immediateNotify=True)

    with cond:
        print("Waiting")
        if not notified[0]:
            cond.wait()

if config.getboolean('CONNECT_TO_SERVER'):
    table = connect()


#test with orange
#lower_color = np.array([0, 180, 200], np.uint8)
#upper_color = np.array([30, 255, 255], np.uint8)

#test with DARK blue
lower_color = np.array([100, 100, 80], np.uint8)
upper_color = np.array([140, 255, 255], np.uint8)

kernel = np.ones((x1,x1), np.uint8)
kernel2 = np.ones((x2,x2), np.uint8)

kernel = np.ones((1,8), np.uint8)

def cv2str(str, x,y):
    cv2.putText(drawing, str,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 2)

def remove_ident(part):
    area.pop(part)
    cXarr.pop(part)
    cYarr.pop(part)
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
    area = []
    t = 0
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cXarr.append(cX)
        cYarr.append(cY)
        area.append(cv2.contourArea(c))
        # draw the contour and center of the shape on the image
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        if t % 2 == 0:
            changing_color = (255,0,255)
        else:
            changing_color = (0,255,0)
        t += 1

        cv2.circle(drawing, (cX, cY), 7, changing_color, -1)

        cv2.putText(drawing, str(cX), (cX - 20, cY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.putText(drawing, str(cY), (cX - 20, cY + 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 2)

    t = 0
    while(t < len(area)):
        if area[t] < min_area:
            remove_ident(t)
        else:
            t+=1
    if len(cXarr)==2: #we have two contours
        #draw the line
        delta_x = abs(cXarr[0]-cXarr[1])
        delta_y = abs(cYarr[0]-cYarr[1])
        cv2.line(drawing,(cXarr[0],cYarr[0]),(cXarr[1],cYarr[1]),(255,0,0),1)
        cv2str(str(area[0]) + " , " + str(area[1]),200, 15)

        if (math.tan(delta_y / delta_x))>0.17:
            cv2str("TOO DAMN MUCH! - deg: " + str(math.tan(delta_y / delta_x) * 180/(math.pi)),20,200)
            value = 3
        else:
            cv2str("you're chillin - deg: " + str(math.tan(delta_y / delta_x) * 180/(math.pi)),20,200)
            value = 1

    elif len(cXarr)==0:
        movement = 3
        delta_x = 0
        delta_y = 0

    else:
        delta_x = 0
        delta_y = 0
        cv2str(do(2),0,0)

    cv2str(str(delta_x)+ " , " + str(delta_y),10,15)

    table.putNumber("mode", value)

    cv2.imshow("raw_image", frame)
    cv2.imshow("mask1", image)
    cv2.imshow("drawing", drawing)

    if config.getboolean('CONNECT_TO_SERVER'):
        table.putNumber("mode", value)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()