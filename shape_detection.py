import cv2
import numpy as np
import cv2.cv as cv
import math
from matplotlib import pyplot as plt


def get_contour_area(contour):
    area = cv2.contourArea(contour)    
    return area

input = cv2.imread('shapes.jpg');
gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
blank_image = np.ones((input.shape[0],input.shape[1],3))
kernel = np.ones((3,3), np.uint8)
cv2.imshow('original', input)

closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Closing',closing)

edged = cv2.Canny(closing,50,90)
cv2.imshow('Canny',edged)

contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours found = "+ str(len(contours)))


cv2.drawContours(input, contours, -1, (0,255,0), 3)

# sort the contours based on area from large to small
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=False)
    
for c in sorted_contours:
    accuracy = 0.03*cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, accuracy, True)
        
    if len(approx) == 3:
        name = "Trigle"
        cv2.drawContours(blank_image,[c],0,(0,255,0),-1)
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(blank_image, name, (cx-25,cy), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
  
    elif len(approx) == 4:
        x,y,w,h = cv2.boundingRect(c)
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        if abs(w-h) <= 3:
            name = "Square"
            cv2.drawContours(blank_image,[c],0,(0,125,255),-1)
            cv2.putText(blank_image, name, (cx-25,cy), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
        else:
            name = "Rect"
            p1 =  approx[0]
            p2 =  approx[1]
            p3 = approx[2]            
            p4 = approx[3]
            # to determine whether its a rectangle or trapezoid
            # we select the two x co-ordinates whose y values are the same
            # and calculate the difference in them. If its a rectangle
            # the difference in the two widths would be close to 0
            # for trapezoid the widths will be different. This can be extended
            # to be more robust like differentiate between the diiferent trapezoids
            if abs(p1[0][1])-abs(p2[0][1]) <= 2:
                length_one = abs(p1[0][0])-abs(p2[0][0])
                length_two = abs(p3[0][0]) - abs(p4[0][0])
            else:
                length_one = abs(p1[0][0])-abs(p3[0][0])
                length_two = abs(p2[0][0]) - abs(p4[0][0])
           
            if abs(length_one) - abs(length_two) <= 1:
                name = "Rect"
            else:
                name = "Trapezoid"
            cv2.drawContours(blank_image,[c],0,(0,0,100),-1)
            M = cv2.moments(c)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(blank_image, name, (cx-25,cy), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    elif len(approx) == 5:
          name = "Pentgon"
          cv2.drawContours(blank_image,[c],0,(125,0,255),-1)
          M = cv2.moments(c)
          cx = int(M['m10'] / M['m00'])
          cy = int(M['m01'] / M['m00'])
          cv2.putText(blank_image, name, (cx-25,cy), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)

    elif len(approx) == 6:
          name = "Hexgon"
          cv2.drawContours(blank_image,[c],0,(0,90,10),-1)
          M = cv2.moments(c)
          cx = int(M['m10'] / M['m00'])
          cy = int(M['m01'] / M['m00'])
          cv2.putText(blank_image, name, (cx-25,cy), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
          
    elif len(approx) > 6  and len(approx) < 10:
          x,y,w,h = cv2.boundingRect(c)
          circles = cv2.HoughCircles(gray, cv.CV_HOUGH_GRADIENT, 1.5, 1)
                    
          if circles is not None:
              circles = np.round(circles[0, :]).astype("int")
              for (a,b,r) in circles:
                  # if its a circle the width of the bounding rectangle
                  # will be twice the radius of the circle
                  if abs(w - abs(2*r)) <= 1:
                      name = "Circle"
                      cv2.drawContours(blank_image,[c],0,(50,50,0),-1)
                  elif abs(w-h) > 10:
                      # if its an ellipse the difference between width and
                      # height of the bounding rectangle will be large
                      name = "Ellipse"
                      cv2.drawContours(blank_image,[c],0,(255,255,0),-1)
                  else:
                      name = "Octagen"
                      cv2.drawContours(blank_image,[c],0,(255,255,0),-1)       
          M = cv2.moments(c)
          cx = int(M['m10'] / M['m00'])
          cy = int(M['m01'] / M['m00'])
          cv2.putText(blank_image, name, (cx-30,cy), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
          
    elif len(approx) == 10:        
          name = "Star"
          cv2.drawContours(blank_image,[c],0,(120,0,0),-1)
          M = cv2.moments(c)
          cx = int(M['m10'] / M['m00'])
          cy = int(M['m01'] / M['m00'])
          cv2.putText(blank_image, name, (cx-20,cy), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)         
          
    cv2.imshow('Draw Large contours only', blank_image)        
    cv2.waitKey() 
                    
cv2.destroyAllWindows()


