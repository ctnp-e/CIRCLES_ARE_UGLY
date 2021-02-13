import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
#cap.set(cv2.CAP_PROP_EXPOSURE, -3)
x1 = 3
x2 = 8
kernel = np.ones((x1,x1), np.uint8)
kernel2 = np.ones((x2,x2), np.uint8)

while True:
    frame = cap.read()[1]
    blurred = cv2.medianBlur(frame, 25)
    x_size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    y_size = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    black = np.zeros((y_size, x_size, 3), np.uint8)

    mask1 = cv2.inRange(cv2.cvtColor(
        blurred, cv2.COLOR_BGR2HSV), (100, 150, 0), (140, 255, 255))
    
    mask1 = cv2.erode(mask1, kernel)
    mask2 = cv2.dilate(mask1,kernel2)

    closed = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel2)

    image = cap.read()[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 11)

    # Morph open 

    # Find contours and filter using contour area and aspect ratio
    cnts = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        area = cv2.contourArea(c)
        if len(approx) > 5 and area > 1000 and area < 500000:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)

    cv2.imshow('image', image)

    cv2.imshow("raw_image", frame)
    #cv2.imshow("mask1", mask1)
    cv2.imshow("mask2", mask2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()