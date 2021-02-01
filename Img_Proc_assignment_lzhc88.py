import numpy as np
import cv2
import random
import math
from scipy.interpolate import UnivariateSpline
import sys

def problem1(img_name,darkening_coef,blending_coef,mode):
    #read an image from the specified file
    #Later change this to arbitrary input image file
    img = cv2.imread(img_name,cv2.IMREAD_COLOR)
    if not img is None:
        rows,cols,channels = img.shape
        #Make image darker
        for x in range(rows):
            for y in range(cols):
                for c in range(channels):
                    img[x,y,c] = darkening_coef*img[x,y,c]
        #############CREATE MASK##########################
        #Do some image thresholding to find the dark areas of the face
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,thresh1 = cv2.threshold(gray,20,255,cv2.THRESH_BINARY)
        ret,thresh2 = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
        ret,thresh3 = cv2.threshold(gray,80,255,cv2.THRESH_BINARY)
        #Find the contours of the face
        whole_row=0
        end_of_face=0
        first=0
        for x in range(int(0.85*rows),int(0.6*rows),-1):
            for y in range(int(0.58*cols),int(0.6*cols)):
                if thresh3[x,y]==0:
                    whole_row=1
                else:
                    whole_row=0
                    break
            if whole_row==1 and first==0:
                end_of_face=x
                first=1
            elif whole_row==1:
                first+=1
        #create mask, ray of light
        mask=np.zeros((rows, cols, channels))
        if mode=='rainbow':
            ray_range=100
            ray_width=25
        else:
            ray_range=60#from which column to which column the ray spans
            ray_width=25#width of ray
        go_right=rows//ray_range#counter to update variable factor, which simulates diagonality of ray
        factor=0#diagonality of ray
        ray_width_change_rate=rows//(ray_width//2)#normally after x rows, the ray width changes at this rate
        ray_width_change=0#ray width change at the beginning, updates with ray_width_change_rate after x rows
        first_thrash=0
        last_thrash=0
        start_big_right_move=0#at the beginning of the face, ray has to move quite a bit to the right to simulate bending
        start_big_left_move=0#at the end of the face, ray has to move quite a bit to the left to simulate bending
        distance=0
        make_ray_thinner=0
        ###end_of_face is a float, which denotes at which percentage of the image length, the face ends##
        if first>55:
            end_of_face=(end_of_face/400)-0.14
        else:
            end_of_face=(end_of_face/400)-0.08
############CREATE SIMPLE MASK###############################################################
        for x in range(rows):
            ##PLANNING MOVES TO BEND AND MAKE THE RAY SLIMMER AT THE RIGHT MOMENT############
            ##RUNNING COUNTERS TO DETECT THE RIGHT MOMENT####################################
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
            ##############################################################################################
            for y in range(cols):
                i = random.randint(0, 1)#randomness makes the ray look less pixeled when moving diagonally
                if y in range((cols//2+20)-(ray_range//2)+factor-i, (cols//2+20)-(ray_range//2)+(ray_width-ray_width_change)+factor-i):#range of ray, pixels that have to be white
                    ########################################################
                    ##MECHANISM TO MAKE RAY THINNER AFTER X ITERATIONS######
                    ##ONLY STARTS AFTER end_of_face PERCENT OF THE FACE#####
                    ##+ABOUT 10 MORE PIXELS#################################
                    if x > ((rows*end_of_face)+10) and make_ray_thinner < 6:
                        make_ray_thinner += 1
                        if make_ray_thinner % 3 == 0:
                            ray_width_change += 1
                    ##########################################################
                    ##PLANNING MOVES FOR BENDING THE RAY AT THE RIGHT MOMENT##
                    ##AN MAKING THE RAY SLIMMER###############################
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
                ##################################################################
                ######COLOR THE PIXELS OF THE MASK################################
                ######THe BRIGHTNESS OF THE PIXELS IN THE RAY DEPENDS ON THE #####
                ######THRASHING RESULT############################################
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
                ####################################################################
        mask = cv2.blur(mask, (7, 7))#blurring the light ray removes the pixeldness of the ray
        #cv2.imshow('Mask',mask)
################CREATE RAINBOW MASK###########################################################
        if mode=='rainbow':
            #####SAME AS BEFORE, BUT WITH COLORS#############################################
            mask2=np.zeros((rows, cols, channels))
            ray_range=100#from which column to which column the ray spans
            go_right=rows//ray_range
            factor=0
            ray_width=60#RAY_WIDTH HAS ALSO CHANGED TO BE WIDER##
            ray_width_change_rate=rows//(ray_width//2)
            ray_width_change=0
            first_thrash=0
            last_thrash=0
            start_big_right_move = 0
            start_big_left_move = 0
            distance = 0
            make_ray_thinner = 0
            ##EXTRA VARIABLES FOR COLOURS#######################################################
            color_change_rate = 4
            colors = [[255, 127, 127], [255, 0, 127], [255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 255,255], [0, 127, 200]]
            #####################################################################################
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
                        mask2[x, y] = colors[col_change]
                        count += 1
                        color_change_rate=4
                    else:
                        mask2[x, y] = [0, 0, 0]
            ##BLUR BOTH MASKS TO MAKE THE RAY LOOK LESS PIXELED########
            mask2=cv2.blur(mask2,(7,7))
            mask = cv2.blur(mask, (7, 7))
            ############################################################
            ##APPLY LESS INTENSITY TO BOTH MASKS########################
            ##LIKE ADDWEIGHTED#########################################
            for x in range(rows):
                for y in range(cols):
                    for c in range(channels):
                        mask[x,y,c] = 0.6*mask[x,y,c]
            for x in range(rows):
                for y in range(cols):
                    for c in range(channels):
                        mask2[x,y,c] = 0.4*mask2[x,y,c]
            ##ADD RAINBOW AND SIMPLE MASK#####################
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
                n = random.random()
                if n<0.5:
                    noise[x,y]=255
                else:
                    noise[x,y]=0
        noise = noise.astype(np.uint8)
        
        #Apply motion blur
        ##CREATE A KERNEL#############################
        kernel = np.zeros((21,21))
        kernel[:,int(len(kernel-1)/2)]=np.ones(len(kernel))
        kernel[int(len(kernel-1)/2),:]=np.ones(len(kernel))
        kernel/=(2*len(kernel))
        #Apply kernel
        noise = cv2.filter2D(noise,-1,kernel)
        ######################################################
        ##VERSION WITHOUT USING cv2.filter2D() FOR CONVOLUTION##
        ##TO ENHANCE THE RUNTIME, DID NOT USE A KERNEL########
        #noise2=np.copy(noise)
        #b=0
        #for x in range(rows):
        #    for y in range(cols):
        #        if y>=cols-4:
        #            b+=1
        #        s=0
        #        for i in range(-4,5-b):
        #            s+=noise[x,y+i]*(1/9)
        #        noise2[x,y]=s
        #    b=0
        #noise = noise2
        #noise = noise.astype(np.uint8)
        #cv2.imshow('Blur noise',noise)
        #####################################################
        if mode=='coloured pencil':
            b=np.copy(noise)
            noise = np.zeros(img.shape)
            for x in range(rows):
                for y in range(cols):
                    noise[x,y]=random.randint(0, 255)
            #Apply motion blur
            ##CREATE A KERNEL#############################
            size = 5
            kernel = np.zeros((size,size))
            for i in range(size):
                for j in range(size):
                    if i==j:
                        kernel[i][j]=1
            kernel/=(len(kernel))
            #Apply kernel
            noise = cv2.filter2D(noise,-1,kernel)
            ##################################################################
            ##SAME OPERATION WITHOUT cv2.filter2D() USED FOR THE CONVOLUTION##
            #noise2=np.copy(noise)
            #cv2.imshow('Noise 2',noise)
            #for x in range(rows):
            #    for y in range(cols):
            #        s=0
            #        for i in range(-6,7):
            #            xs = x+i
            #            if xs>=rows:
            #                xs=rows-1
            #            ys = y+i
            #            if ys>=cols:
            #                ys=cols-1
            #            s+=noise[xs,ys]*(1/13)
            #        noise2[x,y]=s
            #noise = noise2
            ##########################################
            noise = noise.astype(np.uint8)
            g=noise
            ###########################################
            ##MAKE NOISE A BIT DARKER TO MATCH NOISE1##
            a = 0.95
            for x in range(rows):
                for y in range(cols):
                    g[x,y] = np.clip(a*g[x,y],0,255)
            #############################################
            ##BLEND THE NOISE WITH THE GRAY IMAGE########
            for x in range(rows):
                for y in range(cols):
                    gray[x,y] = blending_coef*gray[x,y]
            for x in range(rows):
                for y in range(cols):
                    blend = (1-blending_coef)
                    b[x,y] = blend*b[x,y]
            for x in range(rows):
                for y in range(cols):
                    blend = (1-blending_coef)
                    g[x,y] = blend*g[x,y]
            b=gray+b
            g=gray+g
            ##LASTLY MERGE THE NOISE-IMAGES WITH THE GRAY IMAGE##
            img = cv2.merge((b,g,gray))
        else:
            #############################################
            ##BLEND THE NOISE WITH THE GRAY IMAGE########
            for x in range(rows):
                for y in range(cols):
                    img[x,y] = blending_coef*img[x,y]
            for x in range(rows):
                for y in range(cols):
                    noise[x,y] = (1-blending_coef)*noise[x,y]
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
            else:
                i=abs(i)
                j=abs(j)
                n=math.sqrt((i*i)+(j*j))
            a=gaussian(n,sigma)
            s+=gaussian(n,sigma)
            lst.append(a)
        kernel.append(lst)
    for i in range(-d,d+1):
        for j in range(-d,d+1):
            kernel[i][j]=kernel[i][j]/s
    return kernel

def create_bilinear(sigma,sigma2,kernel_dim,img,x,y):
    rows,cols=img.shape
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
            else:
                n2 = int(img[x,y])-int(img[xs,ys])
                i=abs(i)
                j=abs(j)
                n=math.sqrt((i*i)+(j*j))
            a=gaussian(n,sigma)
            b=gaussian(n2,sigma2)
            s+=a*b
            lst.append(a*b)
        kernel.append(lst)
    for i in range(-d,d+1):
        for j in range(-d,d+1):
            kernel[i][j]=kernel[i][j]/s
    return kernel




def problem3(img_name,blur_amount):
    #filter that first smooths out an image
    #and then applies colour grading
    img = cv2.imread(img_name,cv2.IMREAD_COLOR)
    if not img is None:
        rows,cols,channels = img.shape
        img_cpy=img.copy()
        ######ALTERNATIVELY APPLY GAUSSIAN FILTER#######################################
        ######THIS IS MUCH FASTER THAN BILINEAR FILTER, BUT NOT AS NICE WITH THE EDGES##
        #sigma=blur_amount
        #kernel_dim=5
        #gaussian_kernel=create_gaussian(sigma,kernel_dim)
        #d=math.ceil(kernel_dim/2)
        #for x in range(rows-d):
        #    for y in range(cols-d):
        #        #apply Gaussian Kernel
        #        s=0
        #        for i in range(-d+1,d):
        #            for j in range(-d+1,d):
        #                s+=img[x+i,y+j]*gaussian_kernel[i+d-1][j+d-1]
        #        img_cpy[x,y]=s
        ######APPLY BILINEAR FILTER####################################################
        ######!!!THIS PART TAKES ABOUT 3 MINUTES FOR IMAGE OF SIZE 400*400#############
        sigma=blur_amount-15
        if sigma<=0:
            sigma = 0.1
        sigma2=blur_amount
        kernel_dim=3
        d=math.ceil(kernel_dim/2)
        b,g,r = cv2.split(img_cpy)
        c_cpy=np.zeros((rows,cols))
        channels = [b,g,r]
        copies=[]
        for c in channels:#channel
            for x in range(rows):
                for y in range(cols):
                    #apply Gaussian Kernel
                    s=0
                    g2 = create_bilinear(sigma,sigma2,kernel_dim,c,x,y)
                    for i in range(-d+1,d):
                        for j in range(-d+1,d):
                            xs = x+i
                            if xs>=rows:
                                xs=rows-1
                            ys = y+j
                            if ys>=cols:
                                ys=cols-1
                            s+=c[xs,ys]*g2[i+d-1][j+d-1]
                    c_cpy[x,y]=s
            copies.append(c_cpy.copy())
        img_cpy = cv2.merge((copies[0],copies[1],copies[2]))
        ###########################################################################
        img_cpy=img_cpy.astype(np.uint8)
        
        #####hardcode a lookup table######
        org_values=[0,5,10,20,50,100,150,200,255]
        new_values=[0,10,15,30,80,130,180,220,255]
        new_values_2=[0,5,8,15,40,90,140,200,255]
        new_values_3=[0,5,12,18,48,110,165,200,255]#beautifies the skin-tone if applied to r channel
        #####apply Univariate spline to lookup table########
        spl = UnivariateSpline(org_values,new_values)
        #spl_2 = UnivariateSpline(org_values,new_values_2)not used
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
        cv2.imshow('Original Image',img)
        cv2.imshow('Smoothed Image',img_cpy)
        #cv2.imwrite('Smoothed_gaussian_0.7.png',img_cpy)
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
        ########CHECK CORRECTNESS OF PARAMETERS#################################
        if radius_swirl>=min(center_x,center_y):
            radius_swirl=min(center_x,center_y)-1
            print("We have changed the radius of the swirl to fit the size of the image.")
        ########################################################################
        ########PRE-FILTERING WITH LOW-PASS FILTER##################
        ########USED create_gaussian(sigma,kernel_dim)##############
        ########THE FUNCTION I CREATED ABOVE########################
        sigma=0.7
        kernel_dim=5
        gaussian_kernel=create_gaussian(sigma,kernel_dim)
        gaussian_kernel=np.array(gaussian_kernel)
        img = img.astype(np.float32)
        img = cv2.filter2D(img,-1,gaussian_kernel)
        ############################################################
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
                        continue
                    img_copy[x,y]=1/((x_2-x_1)*(y_2-y_1))*(f_1+f_2+f_3+f_4)
                    ###############################################      
        img_copy=img_copy.astype(np.uint8)
        cv2.imshow('Swirl Image',img_copy)
        cv2.imwrite('Swirl_img_prefiltering.png',img_copy)
        ####################################################################
        #########Do the same thing for image without prefiltering###########
        #########Necessary for subtraction image############################
        img = original_img.copy()
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
                        continue
                    img_copy[x,y]=1/((x_2-x_1)*(y_2-y_1))*(f_1+f_2+f_3+f_4)
                    ###############################################      
        img = img_copy.copy()
        cv2.imwrite('swirl_image_without_prefiltering.png',img_copy)
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
                        continue
                    img_copy[x,y]=1/((x_2-x_1)*(y_2-y_1))*(f_1+f_2+f_3+f_4)
                    ###############################################      
        img_copy=img_copy.astype(np.uint8)
        cv2.imshow('Image warping undone',img_copy)
        cv2.imwrite('warping_undone.png',img_copy)
        subtraction_img = original_img-img_copy
        cv2.imshow('Difference between original and unswirled img',subtraction_img)
        cv2.imwrite('subtraction_image.png',subtraction_img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

#problem1('./face1.jpg',0.6,0.5,'simple')
#problem2('./face1.jpg',0.5,'simple')
#problem2('./face1.jpg',0.5,'coloured pencil')
#problem3('./face2.jpg',30)
#problem4('./face2.jpg',-0.4,150)
#lowpassfilter()
def is_floatstring(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
def is_intstring(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
def main():
    if len(sys.argv)<=1:
        print("No functions were specified, please run the file again specifying which function to run.")
        return
    else:
        if sys.argv[1]=='problem1':
            #run problem1
            if len(sys.argv)>=2:
                #read image file path as string
                img_name = sys.argv[2]
                if len(sys.argv)!=6:
                    print("The number of parameters given does not match the problem.")
                    print("Problem 1 will continue with the default parameters of:")
                    print("darkening coefficient: 0.8")
                    print("blending coefficient: 0.7")
                    print("mode: simple")
                    problem1(img_name,0.8,0.7,'simple')
                else:
                    darkening_coef = sys.argv[3]
                    if is_floatstring(darkening_coef)==False:
                        print("The input for the darkening coefficient could not be turned into a float, ")
                        print("hence the application will proceed with default 0.8")
                        darkening_coef=0.8
                    else:
                        darkening_coef=float(darkening_coef)
                    blending_coef = sys.argv[4]
                    if is_floatstring(blending_coef)==False:
                        print("The input for the blending coefficient could not be turned into a float, ")
                        print("hence the application will proceed with default 0.7")
                        blending_coef=0.7
                    else:
                        blending_coef=float(blending_coef)
                    mode = sys.argv[5]
                    if type(mode)!=str:
                        print("The input for the mode parameter was not a string, ")
                        print("hence the application will proceed with default 'simple'")
                        mode = 'simple'
                    problem1(img_name,darkening_coef,blending_coef,mode)
            else:
                print("Not enough arguments: Please provide an image file as a string as a second command line argument")
                print("e.g. './face1.jpg'")
                return
        elif sys.argv[1]=='problem2':
            #run problem2
            if len(sys.argv)>=2:
                #read image file path as string
                img_name = sys.argv[2]
                if len(sys.argv)!=5:
                    print("The number of parameters given does not match the problem.")
                    print("Problem 2 will continue with the default parameters of:")
                    print("blending coefficient: 0.5")
                    print("mode: simple")
                    problem2(img_name,0.5,'simple')
                else:
                    blending_coef = sys.argv[3]
                    if is_floatstring(blending_coef)==False:
                        print("The input for the blending coefficient was neither a float or int, ")
                        print("hence the application will proceed with default 0.7")
                        blending_coef=0.7
                    else:
                        blending_coef=float(blending_coef)
                    mode = sys.argv[4]
                    if type(mode)!=str:
                        print("The input for the mode parameter was not a string, ")
                        print("hence the application will proceed with default 'simple'")
                        mode = 'simple'
                    problem2(img_name,blending_coef,mode)
            else:
                print("Not enough arguments: Please provide an image file as a string as a second command line argument")
                print("e.g. './face1.jpg'")
                return
        elif sys.argv[1]=='problem3':
            #run problem3
            if len(sys.argv)>=2:
                #read image file path as string
                img_name = sys.argv[2]
                if len(sys.argv)!=4:
                    print("The number of parameters given does not match the problem.")
                    print("Problem 3 will continue with the default parameters of:")
                    print("blur amount: 30")
                    problem3(img_name,30)
                else:
                    blur_amount = sys.argv[3]
                    if is_floatstring(blur_amount)==False:
                        print("The input for the blur amount could not be converted into a float, ")
                        print("hence the application will proceed with default 30")
                        blur_amount=30
                    else:
                        blur_amount=float(blur_amount)
                    problem3(img_name,blur_amount)
            else:
                print("Not enough arguments: Please provide an image file as a string as a second command line argument")
                print("e.g. './face1.jpg'")
                return
        elif sys.argv[1]=='problem4':
            #run problem4
            if len(sys.argv)>=2:
                #read image file path as string
                img_name = sys.argv[2]
                if len(sys.argv)!=5:
                    print("The number of parameters given does not match the problem.")
                    print("Problem 4 will continue with the default parameters of:")
                    print("strength of swirl: -0.4")
                    print("radius of swirl: 150")
                    problem4(img_name,-0.4,150)
                else:
                    strength_swirl = sys.argv[3]
                    if is_floatstring(strength_swirl)==False:
                        print("The input for the strength of the swirl could not be converted into a float, ")
                        print("hence the application will proceed with default -0.4")
                        strength_swirl=-0.4
                    else:
                        strength_swirl=float(strength_swirl)
                    radius_swirl = sys.argv[4]
                    if is_intstring(radius_swirl)==False:
                        print("The input for the radius of the swirl could not be converted into an integer, ")
                        print("hence the application will proceed with default 150")
                        radius_swirl=150
                    else:
                        radius_swirl=float(radius_swirl)
                    problem4(img_name,strength_swirl,radius_swirl)
            else:
                print("Not enough arguments: Please provide an image file as a string as a second command line argument")
                print("e.g. './face1.jpg'")
                return
        else:
            print("Please provide the name of a valid function to execute: ")
            print("Valid names for an executable function are the following: ")
            print("problem1, problem2, problem3, problem4")

if __name__ == "__main__":
    main()



