# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 20:57:33 2020

@author: 123456789
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

time_count=0
count=1
csv_file='data/nyu2_train.csv'
kernel_size=7
sigmaspace=20
sigmacolor=150
csv = open(csv_file, 'r').read()
nyu2_train = list((row.split(',') for row in (csv).split('\n') if len(row) > 0))

# Dataset shuffling happens here
nyu2_train = shuffle(nyu2_train, random_state=0)

# Test on a smaller dataset
filenames = [i[0] for i in nyu2_train]

length = len(filenames)

for filename in filenames:
    apature=torch.Tensor([1.7])
    near=torch.Tensor([1])
    far=torch.Tensor([80])
    pixel_size=torch.Tensor([5.6e-6])
    scale=torch.Tensor([2])
    focal_length=torch.Tensor([0.5])
    focal_depth=torch.Tensor([64])
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
            
    gausdis=-(x*x+y*y).view(1,kernel_size,kernel_size)
    print('gausdis',gausdis.shape)
    padd=torch.nn.ZeroPad2d(kernel_size//2)
    imagepad = image
    imagepad = padd(imagepad)
    I=torch.zeros([3,kernel_size,kernel_size])
    print('pad',imagepad.shape)

    
    for i in range(photo_h):
            #print(i)
            for j in range(photo_w):
                
                I=imagepad[0,:,i:i+kernel_size,j:j+kernel_size]-imagepad[0,:,i+kernel_size//2,j+kernel_size//2].view(3,1,1)
                kernelcore =torch.exp(gausdis/(2*sigmaspace**2)-I**2/(2*sigmacolor**2))
                #print('kernelcore',kernelcore.shape)
                kernelcore_sum=kernelcore.sum(dim=2)
                kernelcore_sum=kernelcore_sum.sum(dim=1)
                kernelcore = kernelcore/(kernelcore_sum).view(3,1,1)
                kerneled=(imagepad[0,:,i:i+kernel_size, j:j+kernel_size] * kernelcore).sum(dim=2)
               # print('kerneled',kerneled.shape)
                image[0,:,i,j]=kerneled.sum(dim=1)
    
    #print('test',imagepad[:,:,0:483,0:640,0,0])#相当于i=6  j=6
   
    #print('imag[:,:,:,:,0,0]',imag[:,:,:,:,0,0])

    #print('kernelcore',kernelcore[0,1,320,320,:,:])
    
    
    end = time.clock()
    print ('time=',end-start)
    outputs=image
    time_count+=end-start
    img = outputs[0]  # plt.imshow()只能接受3-D Tensor，所以也要用image[0]消去batch那一维
    img = torch.clamp(img, 0, 1)
    img = img.numpy()  # FloatTensor转为ndarray
    img = np.transpose(img, (1, 2, 0)) 

    filesave=filename.split('.')

    savefile=filesave[0]+'bila.'+filesave[1]
    print('savefile',savefile)

    io.imsave(savefile,img)
    
print('time_count',time_count)