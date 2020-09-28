# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 09:29:06 2020

@author: 123456789
"""

import torch
#import torch.nn as nn
#from torch.autograd import Function
import skimage.io as io
import cv2
import numpy as np
from sklearn.utils import shuffle
from skimage.transform import resize
#import os
#from zipfile import ZipFile
import time
photo_h=120
photo_w=160
time_count=0
count=1
csv_file='data/nyu2_train.csv'
kernel_size=5
csv = open(csv_file, 'r').read()
nyu2_train = list((row.split(',') for row in (csv).split('\n') if len(row) > 0))

# Dataset shuffling happens here
nyu2_train = shuffle(nyu2_train, random_state=0)

# Test on a smaller dataset
filenames = [i[0] for i in nyu2_train]
labels = [i[1] for i in nyu2_train]
length = len(filenames)

for filename,label in zip(filenames,labels):
    apature=torch.Tensor([1.7])
    near=torch.Tensor([1])
    far=torch.Tensor([80])
    pixel_size=torch.Tensor([5.6e-6])
    scale=torch.Tensor([2])
    focal_length=torch.Tensor([0.5])
    focal_depth=torch.Tensor([64])
    
    image = io.imread(filename)
    depth = io.imread(label)
    image = cv2.resize(image,(photo_w,photo_h),interpolation=cv2.INTER_CUBIC) 
    depth = cv2.resize(depth,(photo_w,photo_h),interpolation=cv2.INTER_CUBIC) 
    image=np.transpose( image,  (2, 0, 1)) 
    depth = torch.Tensor(depth)
    image = torch.Tensor(image)
    image = torch.clamp(image/255.0, 0, 1)
    image = image.view(-1,3,photo_h,photo_w)
    imageori = image
    depth = depth.view(1,photo_h,photo_w)
    Ap = apature.view(-1, 1, 1).expand_as(depth)
    FL = focal_length.view(-1, 1, 1).expand_as(depth)
    focal_depth = focal_depth.view(-1, 1, 1).expand_as(depth)
    Ap = FL / Ap

    real_depth = (far - near) * depth + near
    real_fdepth = (far - near) * focal_depth + near
    c = torch.abs(Ap * (FL * (real_depth - real_fdepth)) / (real_depth * (real_fdepth - FL))) / (pixel_size*scale)
    c = c.clamp(min=1, max=9)
    start = time.clock()

    with torch.no_grad():
            x = torch.arange(kernel_size // 2,
                             -kernel_size // 2,
                             -1).view(kernel_size, 1).float().repeat(1, kernel_size)

            y = torch.arange(kernel_size // 2,
                             -kernel_size // 2,
                             -1).view(1, kernel_size).float().repeat(kernel_size, 1)
            
    weights = c.unsqueeze(1)
    gausdis=-2*(x*x+y*y)

    padd=torch.nn.ZeroPad2d(kernel_size//2)
    imagepad = image
    imagepad = padd(imagepad)
    
  

    
    for i in range(photo_h):
            #print(i)
            for j in range(photo_w):
                kernelcore =2*torch.exp(gausdis/(weights[0,0,i,j]**2))/3.14159265/(weights[0,0,i,j]*weights[0,0,i,j])
                #print('kernelcore',kernelcore.shape)
                kernelcore_sum=kernelcore.sum(dim=1)
                kernelcore_sum=kernelcore_sum.sum(dim=0)
                kernelcore = kernelcore/(kernelcore_sum)
                kerneled=(imagepad[0,:,i:i+kernel_size, j:j+kernel_size] * kernelcore).sum(dim=2)
               # print('kerneled',kerneled.shape)
                image[0,:,i,j]=kerneled.sum(dim=1)
    
    #print('test',imagepad[:,:,0:483,0:photo_w,0,0])#相当于i=6  j=6
   
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

    savefile=filesave[0]+'blur.'+filesave[1]
    print('savefile',savefile)

    io.imsave(savefile,img)
    
print('time_count',time_count)
   