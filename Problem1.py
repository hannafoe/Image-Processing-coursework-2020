import numpy as np
import cv2
import random
def colour_query_mouse_callback(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("BGR colour @ position (%d,%d) = %s" % (x,y, ', '.join(str(i) for i in new_img[y,x])))
    elif  event == cv2.EVENT_RBUTTONDOWN:
        print("BGR colour @ position (%d,%d) = %s" % (x,y, ', '.join(str(i) for i in new_img[y,x])))

#read an image from the specified file
#Later change this to arbitrary input image file
img = cv2.imread('./face1.jpg',cv2.IMREAD_COLOR)
if not img is None:
    #cv2.imshow('Original image',img)
    rows,cols,channels = img.shape
    #Make image darker
    darkening_coef = 0.6 #Change with user input later, between 0 and 1
    for x in range(rows):
        for y in range(cols):
            for c in range(channels):
                img[x,y,c] = darkening_coef*img[x,y,c]
    #cv2.imshow('Darker image',img)
    #Do some image thresholding to find the dark areas of the face
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(gray,20,255,cv2.THRESH_BINARY)
    ret,thresh2 = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
    ret,thresh3 = cv2.threshold(gray,80,255,cv2.THRESH_BINARY)
    cv2.imshow('thresholded image 1',thresh1)
    cv2.imshow('thresholded image 2',thresh2)
    #cv2.imshow('thresholded image 3',thresh3)
    #Find the contours of the face
    edge_img = cv2.Canny(gray,20,255)
    cv2.imshow('Edge image',edge_img)


    #create mask, ray of light
    mask=np.zeros((rows, cols, channels))
    ray_range=60#from which column to which column the ray spans
    go_right=rows//ray_range
    factor=0
    ray_width=30
    ray_width_change_rate=rows//(ray_width//2)
    ray_width_change=0
    first_thrash=0
    start_big_right_move=0
    for x in range(rows):
        if x%go_right==0:
            factor+=1
        if start_big_right_move>0 and start_big_right_move<15:
            start_big_right_move+=1
            factor+=1
        if x%ray_width_change_rate==0:
            ray_width_change+=1
        for y in range(cols):
            i=random.randint(0,1)
            if y in range((cols//2+20)-(ray_range//2)+factor-i,(cols//2+20)-(ray_range//2)+(ray_width-ray_width_change)+factor-i):
                if thresh1[x,y]==[0]:
                    if first_thrash==0:
                        start_big_right_move=1
                        first_thrash=1
                    mask[x,y]=[50,50,50]
                elif thresh2[x,y]==[0]:
                    mask[x,y]=[100,100,100]
                elif thresh3[x,y]==[0]:
                    mask[x,y]=[200,200,200]
                else:
                    mask[x,y]=[250,250,250]
            elif i==1 and (y==(cols//2+20)-(ray_range//2)+factor or y==(cols//2+20)-(ray_range//2)+(ray_width-ray_width_change)+factor):
                mask[x,y]=[200,200,200]
            else:
                mask[x,y] =[0,0,0]
    mask=cv2.blur(mask,(7,7))
    cv2.imshow('Mask',mask)
    img = img.astype(np.uint8)
    mask = mask.astype(np.uint8)
    #img = cv2.addWeighted(img,0.7,mask,0.3,0)
    #Create a function that does the same without using addWeighted
    blending_coef=0.7 #Change with user input later, between 0 and 1, 0 being only the mask and 1 being only the image
    for x in range(rows):
        for y in range(cols):
            for c in range(channels):
                img[x,y,c] = blending_coef*img[x,y,c]
    for x in range(rows):
        for y in range(cols):
            for c in range(channels):
                mask[x,y,c] = (1-blending_coef)*mask[x,y,c]
    #cv2.imshow('Darker image 2',img)
    #cv2.imshow('Mask 2',mask)
    new_img=img+mask
    windowName='New image'
    cv2.namedWindow(windowName)
    cv2.imshow(windowName,new_img)
    cv2.setMouseCallback(windowName,colour_query_mouse_callback)
    cv2.waitKey(0)
else:
    print("No image file successfully loaded.")
cv2.destroyAllWindows()