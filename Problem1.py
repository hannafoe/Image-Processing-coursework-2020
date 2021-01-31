import numpy as np
import cv2
import random
import math
from scipy.interpolate import UnivariateSpline

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
        #Create more sophisticated noise texture
        noise = np.zeros(img.shape)
        for x in range(rows):
            for y in range(cols):
                n = random.random()
                if n<0.5:
                    noise[x,y]=255
                else:
                    noise[x,y]=0



        '''
        #Create noise texture
        noise = np.zeros(img.shape)
        for x in range(rows):
            for y in range(cols):
                noise[x,y]=random.randint(0, 255)'''
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
        b=0
        for x in range(rows):
            for y in range(cols):
                if y>=cols-4:
                    b+=1
                s=0
                for i in range(-4,5-b):
                    s+=noise[x,y+i]*(1/9)
                noise2[x,y]=s
            b=0
        noise = noise2
        noise = noise.astype(np.uint8)
        cv2.imshow('Blur noise',noise)

        
        
        if mode=='coloured pencil':
            b=np.copy(noise)
            noise = np.zeros(img.shape)
            for x in range(rows):
                for y in range(cols):
                    noise[x,y]=random.randint(0, 255)
            noise = noise.astype(np.uint8)
            #noise = cv2.GaussianBlur(noise,(3,3),0)
            noise2=np.copy(noise)
            cv2.imshow('Noise 2',noise)
            for x in range(rows):
                for y in range(cols):
                    s=0
                    for i in range(-6,7):
                        xs = x+i
                        if xs>=rows:
                            xs=rows-1
                        ys = y+i
                        if ys>=cols:
                            ys=cols-1
                        s+=noise[xs,ys]*(1/13)
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
                    blend = ((1-blending_coef))+0.4
                    if blend>=1:
                        blend=1
                        break
                    b[x,y] = blend*b[x,y]
                if blend==1:
                    break
            for x in range(rows):
                for y in range(cols):
                    blend = (1-blending_coef)
                    g[x,y] = blend*g[x,y]
            g=gray+g
            b=gray+b
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

def gaussian(x,sigma):
    oben=math.exp(-0.5*(x**2)/(sigma**2))
    unten=1/(math.sqrt(2*math.pi)*(sigma) )
    return oben*unten
def create_gaussian(sigma,kernel_dim):
    kernel=[]
    d=math.floor(kernel_dim/2)
    s=0
    for i in range(-d,d+1):
        lst=[]
        for j in range(-d,d+1):
            if i==0 and j==0:
                n=0
                #print(n)
            else:
                i=abs(i)
                j=abs(j)
                n=math.sqrt((i*i)+(j*j))
                #print(n)
            a=gaussian(n,sigma)
            s+=gaussian(n,sigma)
            lst.append(a)
        kernel.append(lst)
    for i in range(-d,d+1):
        for j in range(-d,d+1):
            kernel[i][j]=kernel[i][j]/s
    #print(s)
    #print(kernel)
    return kernel

def create_gaussian2(sigma,sigma2,kernel_dim,img,x,y):
    kernel=[]
    d=math.floor(kernel_dim/2)
    s=0
    for i in range(-d,d+1):
        lst=[]
        for j in range(-d,d+1):
            xs = x+i
            if xs>=rows:
                xs=rows-1
            ys = y+j
            if ys>=cols:
                ys=cols-1
            if i==0 and j==0:
                n=0
                n2=0
                #print(n)
            else:
                n2 = int(img[x,y])-int(img[xs,ys])
                #print(n2)
                i=abs(i)
                j=abs(j)
                n=math.sqrt((i*i)+(j*j))
                #print(n)
            a=gaussian(n,sigma)
            b=gaussian(n2,sigma2)
            s+=a*b
            lst.append(a*b)
        kernel.append(lst)
    for i in range(-d,d+1):
        for j in range(-d,d+1):
            kernel[i][j]=kernel[i][j]/s
    #print(s)
    #print(kernel)
    return kernel




def problem3(img_name,blur_amount):
    #filter that first smooths out an image
    #and then applies colour grading
    img = cv2.imread(img_name,cv2.IMREAD_COLOR)
    if not img is None:
        rows,cols,channels = img.shape
        img_cpy=img.copy()
        #gaussian_kernel=[[0.011,0.083,0.011],
        #                    [0.083,0.619,0.083],
        #                    [0.011,0.083,0.011]]
        sigma=8#blur_amount
        sigma2=0.5
        kernel_dim=5
        #gaussian_kernel=create_gaussian(sigma,kernel_dim)
        d=math.ceil(kernel_dim/2)
        b,g,r = cv2.split(img_cpy)
        c_cpy=np.zeros((rows,cols))
        channels = [b,g,r]
        copies=[]
        for c in channels:#channel
            for x in range(rows-d):
                for y in range(cols-d):
                    #apply Gaussian Kernel
                    s=0
                    g2 = create_gaussian2(sigma,sigma2,kernel_dim,c,x,y)
                    for i in range(-d+1,d):
                        for j in range(-d+1,d):
                            s+=c[x+i,y+j]*g2[i+d-1][j+d-1]
                    c_cpy[x,y]=s
            copies.append(c_cpy.copy())
        img_cpy = cv2.merge((copies[0],copies[1],copies[2]))

        img_cpy=img_cpy.astype(np.uint8)
        '''
        #####hardcode a lookup table######
        org_values=[0,5,10,20,50,100,150,200,255]
        new_values=[0,10,15,30,80,130,180,220,255]
        new_values_2=[0,5,8,15,40,80,120,180,255]
        new_values_3=[0,5,12,18,48,110,165,200,255]
        #####apply Univariate spline to lookup table########
        spl = UnivariateSpline(org_values,new_values)
        spl_2 = UnivariateSpline(org_values,new_values_2)
        spl_3=UnivariateSpline(org_values,new_values_3)
        #####Apply new values to img###############
        b,g,r = cv2.split(img_cpy)
        ######Apply new values to channel b########
        for x in range(rows):
            for y in range(cols):
                b[x,y]=spl(b[x,y])
        ######Apply new values to channel g########
        for x in range(rows):
            for y in range(cols):
                g[x,y]=spl(g[x,y])
        ######Apply new values to channel r########
        for x in range(rows):
            for y in range(cols):
                r[x,y]=spl_3(r[x,y])
        img_cpy=cv2.merge((b,g,r))
        cv2.imshow('Original Image',img)'''
        cv2.imshow('Smoothed Image',img_cpy)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def lowpassfilter():
    image = cv2.imread('./face2.jpg',0)
    if not image is None:
        image=image.astype(np.uint8)
        cv2.imshow('Original Image',image)
        rows,cols = image.shape
        center_x = cols//2
        center_y = rows//2
        image = image.astype(np.float32)
        ###pre-filtering, low-pass filtering######
        K=300###cut-off distance (radius) from the Fourier image origin
        fft_img = np.fft.fft2(image)
        fft_img = np.fft.fftshift(fft_img)
        H = np.ones((rows,cols))
        for x in range(rows):
            for y in range(cols):
                #math.sqrt(x**2+y**2)
                #math.exp(-((x-center_x)**2+(y-center_y)**2)/(2*4**2))
                if math.sqrt(x**2+y**2)<=K:
                    H[x,y]=1
                else:
                    H[x,y]=0
        fft_img =H*fft_img
        fft_img=np.fft.ifftshift(fft_img)
        image = np.fft.ifft2(fft_img)
        image = np.real(image)
        image=image.astype(np.uint8)
        cv2.imshow('LPF image',image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def problem4(img_name,strength_swirl,radius_swirl):
    img = cv2.imread(img_name,cv2.IMREAD_COLOR)
    if not img is None:
        img=img.astype(np.uint8)
        original_img = img.copy()
        cv2.imshow('Original Image',img)
        rows,cols,channels = img.shape
        center_x = cols//2
        center_y = rows//2
        '''
        #img = img.astype(np.float32)
        ###pre-filtering, low-pass filtering######
        img_b,img_g,img_r = cv2.split(img)
        images=[img_b,img_g,img_r]
        K=50 ###cut-off distance (radius) from the Fourier image origin
        for image in images:
            fft_img = np.fft.fft2(image)
            fft_img = np.fft.fftshift(fft_img)
            for x in range(rows):
                for y in range(cols):
                    if math.sqrt(x**2+y**2)>K:
                        fft_img[x,y]=0
            fft_img=np.fft.ifftshift(fft_img)
            image = np.fft.ifft2(fft_img)
            image = np.real(image)
        img = cv2.merge((img_b,img_g,img_r))
        img=img.astype(np.uint8)
        cv2.imshow('LPF image',img)'''
        img_copy = img.copy()
        for x in range(rows):
            for y in range(cols):
                norm_x = x-center_x
                norm_y = y-center_y
                r = math.sqrt((norm_x**2)+(norm_y**2)) #distance from center of img
                if norm_x==0:
                    norm_x=0.001
                theta = math.atan(norm_y/norm_x) #angle from center of img
                if x<center_x:
                    theta+=math.pi
                strength_swirl_change = 1-(r/radius_swirl)
                if(strength_swirl_change>0):
                    angle = strength_swirl*strength_swirl_change*math.pi*2
                    theta += angle
                    #####nearest neighbour interpolation######
                    #norm_x = int(r*math.cos(theta)+0.5)
                    #norm_y = int(r*math.sin(theta)+0.5)
                    #img_copy[x,y]= img[norm_x+center_x,norm_y+center_y]
                    ##########################################
                    #####bilinear interpolation#################
                    norm_x = r*math.cos(theta)+center_x
                    norm_y = r*math.sin(theta)+center_y
                    x_1=math.floor(norm_x)
                    y_1=math.floor(norm_y)
                    x_2=math.ceil(norm_x)
                    y_2=math.ceil(norm_y)
                    f_1=img[x_1,y_1]*(x_2-norm_x)*(y_2-norm_y)
                    f_2=img[x_2,y_1]*(norm_x-x_1)*(y_2-norm_y)
                    f_3=img[x_1,y_2]*(x_2-norm_x)*(norm_y-y_1)
                    f_4=img[x_2,y_2]*(norm_x-x_1)*(norm_y-y_1)
                    if x_2==x_1 or y_2==y_1:
                        print(norm_x,norm_y)
                        continue
                    img_copy[x,y]=1/((x_2-x_1)*(y_2-y_1))*(f_1+f_2+f_3+f_4)
                    ###############################################      
        img_copy=img_copy.astype(np.uint8)
        cv2.imshow('Swirl Image',img_copy)
        img = img_copy.copy()
        #cv2.imwrite('swirl_image_bilin.png',img_copy)
        #######image warp inverse transformation######
        for x in range(rows):
            for y in range(cols):
                norm_x = x-center_x
                norm_y = y-center_y
                r = math.sqrt((norm_x**2)+(norm_y**2)) #distance from center of img
                if norm_x==0:
                    norm_x=0.001
                theta = math.atan(norm_y/norm_x) #angle from center of img
                if x<center_x:
                    theta+=math.pi
                strength_swirl_change = 1-(r/radius_swirl)
                if(strength_swirl_change>0):
                    angle = -strength_swirl*strength_swirl_change*math.pi*2
                    theta += angle
                    #####nearest neighbour interpolation######
                    #norm_x = int(r*math.cos(theta)+0.5)
                    #norm_y = int(r*math.sin(theta)+0.5)
                    #img_copy[x,y]= img[norm_x+center_x,norm_y+center_y]
                    ##########################################
                    #####bilinear interpolation#################
                    norm_x = r*math.cos(theta)+center_x
                    norm_y = r*math.sin(theta)+center_y
                    x_1=math.floor(norm_x)
                    y_1=math.floor(norm_y)
                    x_2=math.ceil(norm_x)
                    y_2=math.ceil(norm_y)
                    f_1=img[x_1,y_1]*(x_2-norm_x)*(y_2-norm_y)
                    f_2=img[x_2,y_1]*(norm_x-x_1)*(y_2-norm_y)
                    f_3=img[x_1,y_2]*(x_2-norm_x)*(norm_y-y_1)
                    f_4=img[x_2,y_2]*(norm_x-x_1)*(norm_y-y_1)
                    if x_2==x_1 or y_2==y_1:
                        print(norm_x,norm_y)
                        continue
                    img_copy[x,y]=1/((x_2-x_1)*(y_2-y_1))*(f_1+f_2+f_3+f_4)
                    ###############################################      
        img_copy=img_copy.astype(np.uint8)
        cv2.imshow('Back to original',img_copy)
        subtraction_img = original_img-img_copy
        #subtraction_img=cv2.subtract(original_img,img_copy)
        cv2.imshow('Difference between original and unswirled img',subtraction_img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

#problem1('./face1.jpg',0.6,0.5,'simple')
#problem2('./face1.jpg',0.5,'simple')
#problem2('./face1.jpg',0.5,'coloured pencil')
problem3('./face1.jpg',0.7)
#problem4('./face2.jpg',-0.4,150)
#lowpassfilter()
