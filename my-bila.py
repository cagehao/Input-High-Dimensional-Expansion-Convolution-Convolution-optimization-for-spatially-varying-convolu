# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 21:38:19 2020

@author: 123456789
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 18:46:52 2020

@author: 123456989
"""
import cv2
import torch
#import torch.nn as nn
#from torch.autograd import Function
import skimage.io as io

import numpy as np
from sklearn.utils import shuffle
#import os
#from zipfile import ZipFile
 
import time
photo_h=480
photo_w=640
sigmaspace=20
sigmacolor=150
time_count=0
count=1
csv_file='data/nyu2_train.csv'
kernel_size=7
csv = open(csv_file, 'r').read()
nyu2_train = list((row.split(',') for row in (csv).split('\n') if len(row) > 0))

# Dataset shuffling happens here
nyu2_train = shuffle(nyu2_train, random_state=0)

# Test on a smaller dataset
filenames = [i[0] for i in nyu2_train]
length = len(filenames)

for filename in filenames:
    
    
    image = io.imread(filename)
    image = cv2.resize(image,(photo_w,photo_h),interpolation=cv2.INTER_CUBIC) 
    image=np.transpose( image,  (2, 0, 1)) 
   
    image = torch.Tensor(image)
    image = torch.clamp(image/255.0, 0, 1)
    image = image.view(-1,3,photo_h,photo_w)
    imageori = image
   

    start = time.clock()

    with torch.no_grad():
            x = torch.arange(kernel_size // 2,
                             -kernel_size // 2,
                             -1).view(kernel_size, 1).float().repeat(1, kernel_size)

            y = torch.arange(kernel_size // 2,
                             -kernel_size // 2,
                             -1).view(1, kernel_size).float().repeat(kernel_size, 1)
    
    

    imag = image
    imag = imag.view(1, 3, photo_h, photo_w, 1, 1)
    I=imag.expand(1, 3, photo_h, photo_w, kernel_size, kernel_size)
    padd=torch.nn.ZeroPad2d(kernel_size//2)
    imagepad = image
    imagepad = padd(imagepad)
    imagepad = imagepad.view(1, 3, photo_h+kernel_size-1, photo_w+kernel_size-1, 1, 1)
    imagepad = imagepad.expand(1, 3, photo_h+kernel_size-1, photo_w+kernel_size-1, kernel_size, kernel_size)
    #print('pad',imagepad.shape)
    
    imag = imag.expand(1, 3, photo_h, photo_w, kernel_size, kernel_size).clone()
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            imag[:,:,:,:,i,j] = imagepad[:,:,i:(photo_h+i),j:(photo_w+j),0,0]
            
    
    #print('test',imagepad[:,:,0:483,0:photo_w,0,0])#相当于i=6  j=6
   
    #print('imag[:,:,:,:,0,0]',imag[:,:,:,:,0,0])
    
    x = x.view(1, 1, 1, 1, kernel_size, kernel_size)
    y = y.view(1, 1, 1, 1, kernel_size, kernel_size)
    x = x.expand(1, 3, photo_h, photo_w, kernel_size, kernel_size)
    y = y.expand(1, 3, photo_h, photo_w, kernel_size, kernel_size)
    I=imag-I
    gausdis=-(x*x+y*y)
    print('gausdis',gausdis.shape)
    kernelcore = torch.exp(gausdis/(2*sigmaspace**2)-I**2/(2*sigmacolor**2))
    weightsum = kernelcore.sum(dim=5)
    weightsum = weightsum.sum(dim=4)
    weightsum = weightsum.view(1, 3, photo_h, photo_w, 1, 1)
    weightsum = weightsum.expand(1, 3, photo_h, photo_w, kernel_size, kernel_size)
    kernelcore = kernelcore/weightsum
   # print('kernelcore',kernelcore)
    #print('kernelcore',kernelcore[0,1,320,320,:,:])
    blurrgb=kernelcore*imag
    blurrgb=blurrgb.sum(dim=5)
    outputs=blurrgb.sum(dim=4)
    end = time.clock()
    print ('time=',end-start)
    time_count+=end-start
    
    img = outputs[0]  # plt.imshow()只能接受3-D Tensor，所以也要用image[0]消去batch那一维
    img = torch.clamp(img, 0, 1)
    img = img.numpy()  # FloatTensor转为ndarray
    img = np.transpose(img, (1, 2, 0)) 

    filesave=filename.split('.')

    savefile=filesave[0]+'bila.'+filesave[1]
    print('savefile',savefile)

    io.imsave(savefile,img)
    count=count+1


print('time_count',time_count)