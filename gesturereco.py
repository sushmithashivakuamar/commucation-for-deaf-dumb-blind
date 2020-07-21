#!/usr/bin/env python
import cv2
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
time.sleep(3)
error =0
lower_thresh1 = 129 
upper_thresh1 = 255
PI = math.pi
j=0
while(cap.isOpened()):

    ret, img = cap.read()

    
    cv2.rectangle(img, (60,60), (300,300), (255,255,255),4)
    crop_img = img[70:300, 70:300]
    crop_img1 = img[70:300, 70:300]
    
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0,150,50])

    upper_red = np.array([195,255,255])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(img,img, mask= mask)   

    
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)

    

    _, thresh1 = cv2.threshold(blurred, lower_thresh1, upper_thresh1, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow('Thresholded', thresh1)
    (contours, hierarchy) = cv2.findContours(thresh1.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
   
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    area_of_contour = cv2.contourArea(cnt)

    
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

    
    hull = cv2.convexHull(cnt)

    
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

    
    hull = cv2.convexHull(cnt, returnPoints=False)

    
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)


    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]

        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57


        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img, far, 1, [0,0,255], -1)
        dist = cv2.pointPolygonTest(cnt,far,True) #---
        

        cv2.line(crop_img,start, end, [0,255,0], 2)
        moment = cv2.moments(cnt)   
        perimeter = cv2.arcLength(cnt,True)
        area = cv2.contourArea(cnt)


        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(crop_img,center,radius,(255,0,0),2)

        area_of_circle=PI*radius*radius



        hull_test = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull_test)
        solidity = float(area)/hull_area


        aspect_ratio = float(w)/h

        rect_area = w*h
        extent = float(area)/rect_area

        (x,y),(MA,ma),angle_t = cv2.fitEllipse(cnt)
        cv2.circle(crop_img,far,5,[0,0,255],-1) 

        if area_of_circle - area <5000:


            cv2.putText(img,"AAAAAA ", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            
    error=1
  
    if count_defects == 1:

        if angle_t < 10:

             cv2.putText(img,"VVVVVVVVVV ", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

        if 40 < angle_t < 66:

             cv2.putText(img,"CCCC ", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        if 20 < angle_t < 35:

             cv2.putText(img,"lllllllllllll ", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

    elif count_defects == 2:

        if angle_t > 100:

            
             str = "FFFFFFFFFFFFFFFFFFFFFFFFF"
             cv2.putText(img, str, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        else:
            str = "WWWWWWWWWWWWWWWWWWWWWW"
            cv2.putText(img, str, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    elif count_defects == 3:
        
        cv2.putText(img,"BBBBB", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 4:
        cv2.putText(img,"uuuu", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

    else :
        if area > 12000:

            cv2.putText(img,"BBBBBBBBBBBBBBBBBBBB", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        else:
            if solidity < 0.85:
                if aspect_ratio < 1:
                     if angle_t < 20:

                         cv2.putText(img,"DDDDD", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

                     elif 169<angle_t <180:

                         cv2.putText(img,"IIIIIIIII", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                     elif angle_t < 168:

                         cv2.putText(img,"JJJJJJJJJJJJJJJ", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                elif aspect_ratio > 1.01:
                     cv2.putText(img,"yyyyyyyyyyy", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                else:
                    if angle_t > 30 and angle_t < 100:
                        cv2.putText(img,"HHHHHHHHHHHHHH", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                        
   
    cv2.imshow('Gesture', img)
    all_img = np.hstack((drawing, crop_img))
    cv2.imshow('Contours', all_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if  error==0:
   print('Camera interupted \nPlease exeute the script again') 

