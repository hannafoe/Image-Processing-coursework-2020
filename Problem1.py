import numpy as np
import cv2

#read an image from the specified file
#Later change this to arbitrary input image file
img = cv2.imread('./face1.jpg',cv2.IMREAD_COLOR)
if not img is None:
    cv2.imshow('Original image',img)
    rows,cols,channels = img.shape
    #Make image darker
    darkening_coef = 0.6 #Change with user input later, between 0 and 1
    for x in range(rows):
        for y in range(cols):
            for c in range(channels):
                img[x,y,c] = darkening_coef*img[x,y,c]
    cv2.imshow('Darker image',img)
    #create mask, ray of light
    mask=np.zeros((rows, cols, channels))
    for x in range(rows):
        for y in range(cols):
            if y in range(180,220):
                mask[x,y]=[250,250,250]
            else:
                mask[x,y] =[0,0,0]
    cv2.imshow('Mask',mask)
    img = img.astype(np.uint8)
    mask = mask.astype(np.uint8)
    #img = cv2.addWeighted(img,0.7,mask,0.3,0)
    #Create a function that does the same without using addWeighted
    blending_coef=0.8 #Change with user input later, between 0 and 1, 0 being only the image and 1 adding the whole mask to the whole image
    for x in range(rows):
        for y in range(cols):
            for c in range(channels):
                mask[x,y,c] = blending_coef*mask[x,y,c]
    cv2.imshow('Darker image 2',img)
    cv2.imshow('Mask 2',mask)
    """
    new_img=np.zeros((rows, cols, channels))
    for x in range(rows):
        for y in range(cols):
            for c in range(channels):
                new_img[x,y,c] = img[x,y,c]+mask[x,y,c]
    """
    new_img=img+mask
    cv2.imshow('New image',new_img)
    cv2.waitKey(0)
else:
    print("No image file successfully loaded.")
cv2.destroyAllWindows()