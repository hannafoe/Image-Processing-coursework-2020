import numpy as np
import cv2
import random

def problem1(img_name,darkening_coef,blending_coef,mode):
    #read an image from the specified file
    #Later change this to arbitrary input image file
    if img_name=='./face1.jpg':
        print("Face 1")
    if img_name=='./face2.jpg':
        print("Face 2")
    img = cv2.imread(img_name,cv2.IMREAD_COLOR)
    if not img is None:
        rows,cols,channels = img.shape
        #Make image darker
        for x in range(rows):
            for y in range(cols):
                for c in range(channels):
                    img[x,y,c] = darkening_coef*img[x,y,c]
        #cv2.imshow('Darker image',img)

        #############CREATE MASK##########################
        #Do some image thresholding to find the dark areas of the face
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,thresh1 = cv2.threshold(gray,20,255,cv2.THRESH_BINARY)
        ret,thresh2 = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
        ret,thresh3 = cv2.threshold(gray,80,255,cv2.THRESH_BINARY)
        #cv2.imshow('thresholded image 1',thresh1)
        #cv2.imshow('thresholded image 2',thresh2)
        #cv2.imshow('thresholded image 3',thresh3)
        #Find the contours of the face
        whole_row=0
        end_of_face=0
        first=0
        for x in range(int(0.85*rows),int(0.6*rows),-1):
            for y in range(int(0.58*cols),int(0.6*cols)):
                if thresh3[x,y]==0:
                    #print(x,y)
                    whole_row=1
                else:
                    whole_row=0
                    break
            if whole_row==1 and first==0:
                end_of_face=x
                first=1
            elif whole_row==1:
                first+=1

        #print(end_of_face)
        #print(first)
        #create mask, ray of light
        mask=np.zeros((rows, cols, channels))
        if mode=='rainbow':
            ray_range=100
            ray_width=25
        else:
            ray_range=60#from which column to which column the ray spans
            ray_width=25
        go_right=rows//ray_range
        factor=0
        ray_width_change_rate=rows//(ray_width//2)
        ray_width_change=0
        first_thrash=0
        last_thrash=0
        start_big_right_move=0
        start_big_left_move=0
        distance=0
        make_ray_thinner=0
        if first>55:
            end_of_face=(end_of_face/400)-0.14
        else:
            end_of_face=(end_of_face/400)-0.08
        #print(end_of_face)
############CREATE SIMPLE MASK###############################################################
        if mode != 'r':
            for x in range(rows):
                if x % go_right == 0:
                    factor += 1
                if start_big_right_move > 0 and start_big_right_move < 30:
                    start_big_right_move += 1
                    if start_big_right_move % 3 == 0:
                        factor += 1
                if start_big_left_move > 0 and start_big_left_move < 20:
                    start_big_left_move += 1
                    if start_big_left_move % 2 == 0:
                        ray_width_change += 1
                elif start_big_left_move >= 20 and start_big_left_move < 30:
                    start_big_left_move += 1
                    if start_big_left_move < 26:
                        factor -= 1
                    ray_width_change -= 1
                if x % ray_width_change_rate == 0:
                    ray_width_change += 1
                for y in range(cols):
                    i = random.randint(0, 1)
                    if y in range((cols//2+20)-(ray_range//2)+factor-i, (cols//2+20)-(ray_range//2)+(ray_width-ray_width_change)+factor-i):
                        if x > ((rows*end_of_face)+10) and make_ray_thinner < 6:
                            make_ray_thinner += 1
                            if make_ray_thinner % 3 == 0:
                                ray_width_change += 1
                        if thresh1[x, y] == [0] or thresh2[x, y] == 0:
                            if first_thrash == 0 and y in range((cols//2+20)-(ray_range//2)+(ray_width-ray_width_change)+factor-5, (cols//2+20)-(ray_range//2)+(ray_width-ray_width_change)+factor):
                                start_big_right_move = 1
                                first_thrash = 1
                            if x > (rows*end_of_face) and last_thrash == 0 and y in range((cols//2+20)-(ray_range//2)+(ray_width-ray_width_change)+factor-5, (cols//2+20)-(ray_range//2)+(ray_width-ray_width_change)+factor):
                                start_big_left_move = 1
                                last_thrash = 1
                            elif x-distance > 10 and x > (rows*0.3) and y in range((cols//2+20)-(ray_range//2)+(ray_width-ray_width_change)+factor-5, (cols//2+20)-(ray_range//2)+(ray_width-ray_width_change)+factor):
                                factor -= 1
                                distance = x
                        if thresh1[x, y] == [0] and (x > (rows*0.6) or x < (rows*0.2)):
                            mask[x, y] = [25, 25, 25]
                        elif thresh2[x, y] == [0] and (x > (rows*0.6) or x < (rows*0.2)):
                            mask[x, y] = [100, 100, 100]
                        elif thresh3[x, y] == [0]:
                            mask[x, y] = [200, 200, 200]
                        else:
                            mask[x, y] = [255, 255, 255]
                    elif i == 1 and (y == (cols//2+20)-(ray_range//2)+factor or y == (cols//2+20)-(ray_range//2)+(ray_width-ray_width_change)+factor):
                        mask[x, y] = [200, 200, 200]
                    else:
                        mask[x, y] = [0, 0, 0]
            mask = cv2.blur(mask, (7, 7))
            #cv2.imshow('Mask',mask)
################CREATE RAINBOW MASK###########################################################
        if mode=='rainbow':
            mask2=np.zeros((rows, cols, channels))
            ray_range=100#from which column to which column the ray spans
            go_right=rows//ray_range
            factor=0
            ray_width=60
            ray_width_change_rate=rows//(ray_width//2)
            ray_width_change=0
            first_thrash=0
            last_thrash=0
            start_big_right_move = 0
            start_big_left_move = 0
            distance = 0
            make_ray_thinner = 0
            color_change_rate = 4
            colors = [[255, 127, 127], [255, 0, 127], [255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 255,255], [0, 127, 200]]
            for x in range(rows):
                if x % go_right == 0:
                    factor += 1
                if start_big_right_move > 0 and start_big_right_move < 30:
                    start_big_right_move += 1
                    if start_big_right_move % 3 == 0:
                        factor += 1
                if start_big_left_move > 0 and start_big_left_move < 20:
                    start_big_left_move += 1
                    if start_big_left_move % 2 == 0:
                        ray_width_change += 1
                elif start_big_left_move >= 20 and start_big_left_move < 30:
                    start_big_left_move += 1
                    if start_big_left_move < 26:
                        factor -= 1
                    ray_width_change -= 1
                if x % ray_width_change_rate == 0:
                    ray_width_change += 1
                col_change = 0
                count = 0
                for y in range(cols):
                    if y in range((cols//2)-(ray_range//2)+factor, (cols//2)-(ray_range//2)+(ray_width-ray_width_change)+factor):
                        if(count % color_change_rate == 0) and col_change!=6:
                            col_change += 1
                            if col_change==4:
                                #print(colors[col_change])
                                color_change_rate=10
                        if x > ((rows*end_of_face)+10) and make_ray_thinner < 6:
                            make_ray_thinner += 1
                            if make_ray_thinner % 3 == 0:
                                ray_width_change += 1
                        if thresh1[x, y] == [0] or thresh2[x, y] == 0:
                            if first_thrash == 0 and y in range((cols//2)-(ray_range//2)+(ray_width-ray_width_change)+factor-5, (cols//2)-(ray_range//2)+(ray_width-ray_width_change)+factor):
                                start_big_right_move = 1
                                first_thrash = 1
                            if x > (rows*end_of_face) and last_thrash == 0 and y in range((cols//2)-(ray_range//2)+(ray_width-ray_width_change)+factor-5, (cols//2)-(ray_range//2)+(ray_width-ray_width_change)+factor):
                                start_big_left_move = 1
                                last_thrash = 1
                            elif x-distance > 10 and x > (rows*0.3) and y in range((cols//2)-(ray_range//2)+(ray_width-ray_width_change)+factor-5, (cols//2)-(ray_range//2)+(ray_width-ray_width_change)+factor):
                                factor -= 1
                                distance = x
                        #print(colors[col_change],col_change,x,y)
                        mask2[x, y] = colors[col_change]
                        count += 1
                        color_change_rate=4
                    else:
                        mask2[x, y] = [0, 0, 0]
            mask2=cv2.blur(mask2,(7,7))
            #cv2.imshow('Mask 2',mask2)
            mask = cv2.blur(mask, (7, 7))
            #cv2.imshow('Mask',mask)
            for x in range(rows):
                for y in range(cols):
                    for c in range(channels):
                        mask[x,y,c] = 0.3*mask[x,y,c]
            for x in range(rows):
                for y in range(cols):
                    for c in range(channels):
                        mask2[x,y,c] = 0.2*mask2[x,y,c]
            mask = mask+mask2
            #cv2.imshow('Mask sum',mask)
 ##########Now mask has been created########################################       
        img = img.astype(np.uint8)
        mask = mask.astype(np.uint8)
        #img = cv2.addWeighted(img,0.7,mask,0.3,0)
        #Create a function that does the same without using addWeighted
        #blending coefficient between 0 and 1, 0 being only the mask and 1 being only the image
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
        cv2.imshow('New image',new_img)
        cv2.waitKey(0)
    else:
        print("No image file successfully loaded.")
    cv2.destroyAllWindows()
    
def problem2(img_name,blending_coef,mode):
    img = cv2.imread(img_name,0)
    gray=img
    if not img is None:
        rows,cols = img.shape

        #Create noise texture
        noise = np.zeros(img.shape)
        for x in range(rows):
            for y in range(cols):
                noise[x,y]=random.randint(0, 255)
        noise = noise.astype(np.uint8)
        cv2.imshow('Noise',noise)
        #Apply motion blur
        #instead of building a kernel and then multiplying convulating the kernel with each pixel of image
        #since this takes a lot of time, only multiply image with elements that wouldn't be zero in the kernel
        #kernel = np.zeros((9,9))
        #kernel[:,int(len(kernel-1)/2)]=np.ones(len(kernel))
        #kernel[:,0]=np.ones(len(kernel))
        #kernel[int(len(kernel-1)/2),:]=np.ones(len(kernel))
        #kernel/=(2*len(kernel))
        #Apply kernel
        noise2=np.copy(noise)
        for x in range(rows-6):
            for y in range(cols-6):
                s=0
                for i in range(-6,7):
                    s+=noise[x,y+i]*(1/13)
                noise2[x,y]=s
        noise = noise2
        noise = noise.astype(np.uint8)
        cv2.imshow('Blur noise',noise)

        
        
        if mode=='coloured pencil':
            b=noise
            noise = np.zeros(img.shape)
            for x in range(rows):
                for y in range(cols):
                    noise[x,y]=random.randint(0, 255)
            noise = noise.astype(np.uint8)
            noise2=np.copy(noise)
            cv2.imshow('Noise 2',noise)
            for x in range(rows-4):
                for y in range(cols-4):
                    s=0
                    for i in range(-4,5):
                        s+=noise[x+i,y+i]*(1/9)
                    noise2[x,y]=s
            noise = noise2
            noise = noise.astype(np.uint8)
            cv2.imshow('Blur noise 2',noise)
            g=noise
            for x in range(rows):
                for y in range(cols):
                    gray[x,y] = blending_coef*gray[x,y]
            for x in range(rows):
                for y in range(cols):
                    b[x,y] = (1-blending_coef)*b[x,y]
            for x in range(rows):
                for y in range(cols):
                    g[x,y] = (1-blending_coef)*g[x,y]
            img = cv2.merge((b,g,gray))
        else:
            for x in range(rows):
                for y in range(cols):
                    img[x,y] = blending_coef*img[x,y]
            for x in range(rows):
                for y in range(cols):
                    noise[x,y] = (1-blending_coef)*noise[x,y]
            #cv2.addWeighted(img,0.5,noise,1,0.0)
            #cv2.add(img,noise)
            img=img+noise
            img = img.astype(np.uint8)
        cv2.imshow('Image',img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


#problem1('./face2.jpg',0.6,0.5,'rainbow')
problem2('./face1.jpg',0.8,'')

