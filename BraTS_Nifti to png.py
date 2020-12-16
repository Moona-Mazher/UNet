# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 19:30:56 2020

@author: moona
"""

#%% 
#how to read a complete list of files in a folder
import nibabel as nib
import cv2
import os
import matplotlib.pyplot as plt
#ef createFolder(directory):
#        if not os.path.exists(directory):
#           os.makedirs(directory)
#   except OSError:
#       print ('Error: Creating directory. ' +  directory)


Path_BraTS_Tr_Nifti='C:\\Users\\moona\\Desktop\\PHD WORK\\BraTS Dataset\\MICCAI_BraTS2020_TrainingData\\Training\\'
Pathlist=os.listdir(Path_BraTS_Tr_Nifti)
Path_Save_BraTS_Tr='C:\\Users\\moona\\Desktop\\PHD WORK\\BraTS Dataset\\Training\\T1CE\\'

for i in Pathlist:
    print(i)
    image_folder=nib.load(os.path.join(Path_BraTS_Tr_Nifti,i+"\\"+i+"_t1ce.nii.gz"))
    image_array=image_folder.get_data()
   # image_array=image_array[:,:,:]
    image_array.shape[2]
#   path1=os.path.join(Path_Save_BraTS_Tr,str(i.split('.')[0]))
#   createFolder(path1)
    for ii in range(0,image_array.shape[2]):
        img1=image_array[:,:,ii]
        cv2.imwrite(Path_Save_BraTS_Tr+str(ii)+str(i.split('.')[0])+'.png',img1)
        print(ii)