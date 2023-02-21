'''
Descripttion: 
version: 
Author: Jinlong Li CSU PhD
Date: 2022-02-21 16:13:51
LastEditors: Jinlong Li CSU PhD
LastEditTime: 2022-02-25 18:48:33
'''
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
import pdb

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 10

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import os


def imshow(img):    
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()


config_file = '/home/jinlong/2.Special_issue_DA/Domain-Adaptive-Faster-RCNN-PyTorch/configs/da_faster_rcnn/e2e_triplet_da_faster_rcnn_R_50_C4_cityscapes_to_foggy_cityscapes.yaml'   #this is replaced by my yaml file 
cfg.merge_from_file(config_file)                            # update the config options with the config file

# cfg.MODEL.WEIGHT = '/home/jinlong/2.Special_issue_DA/trained_models/tri-w=1_m=1_G_new/model_0100000.pth'
#this is the output .pth file
cfg.MODEL.WEIGHT =  '/home/jinlong/2.Special_issue_DA/trained_models/faster_r-cnn/model_final.pth'
# cfg.MODEL.WEIGHT = '/home/jinlong/2.Special_issue_DA/trained_models/Align_img+ins/model_final.pth'
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"]) # manual override some options



coco_demo = COCODemo(
    cfg,
    min_image_size=1024,
    confidence_threshold=0.9,
)


# val_path='/home/jinlong/2.Special_issue_DA/dataset/DA_cityscapes/leftImg8bit_trainvaltest_foggy/leftImg8bit_foggy/val/' #this is the validation image data

val_path='/home/jinlong/2.Special_issue_DA/dataset/DA_cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/' #this is the validation image data

imglistval = os.listdir(val_path) 
for name in imglistval:

    imgfile = val_path + name
    pil_image = Image.open(imgfile).convert("RGB")
    image = np.array(pil_image)[:, :, [2, 1, 0]]

    predictions = coco_demo.run_on_opencv_image(image) # forward predict
    # plt.subplot(1, 2, 1)
    # plt.imshow(image[:,:,::-1])
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    plt.imshow(predictions[:,:,::-1])
    plt.axis('off')
    plt.savefig(os.path.join('/home/jinlong/2.Special_issue_DA/test/clean-noadapt', str(name)))#   noadapt     baseline our
    # plt.show()
    # imshow(predictions)
    # pdb.set_trace() 