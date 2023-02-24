import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn import DataParallel
import torch.nn.functional as F

import numpy as np
import scipy
import cv2
import glob, os
from PIL import Image
import gc
import argparse

from torchvision import transforms
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
from models import ModelBuilder, SegmentationModule

import imageio
import scipy.io
import sys

import warnings
warnings.filterwarnings("ignore")

import mpi_net35
from scipy.ndimage.filters import gaussian_filter
import time

input_file='data/1.avi'
out_dir='result/'
if os.path.exists(out_dir)==False:
    os.mkdir(out_dir)
SEG=300

rho=0.1
nframe=20;
Nkeep=5
batchsize=8
margin=64

builder = ModelBuilder()
net_encoder = builder.build_encoder(
    arch='resnet50dilated',
    fc_dim=2048,
    weights='baseline-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
net_decoder = builder.build_decoder(
    arch='ppm_deepsup',
    fc_dim=2048,
    num_class=150,
    weights='baseline-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
    use_softmax=True)
crit = torch.nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.eval()
normalize = transforms.Normalize(
            mean=[102.9801, 115.9465, 122.7717],
            std=[1., 1., 1.])

def img_transform( img):
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    img = normalize(torch.from_numpy(img.copy()))
    return img

def round2nearest_multiple( x, p):
    return ((x - 1) // p + 1) * p

def mycompute_mask(imgs):
    c = 0
    for img in imgs:
        #inputimg = img.transpose(1,2,0)
        #inputimg = np.concatenate((inputimg,inputimg,inputimg),axis=2)
        img = img.squeeze(0)
        print(img.shape)
        M_img = compute_mask(img)
        M_img = M_img.unsqueeze(0)
        M_img = M_img.unsqueeze(0)
        if c==0:
            mask = M_img
            c = c+1
        else:
            mask = torch.cat([mask,M_img],dim=0)
    #mask = np.concatenate((mask1,mask2), axis=0)
    return mask


def compute_mask(img):
    ori_height, ori_width, _ = img.shape
    imgSize=[300, 400, 500, 600]#
    img_resized_list = []
    for this_short_size in imgSize:
        scale = min(this_short_size / float(min(ori_height, ori_width)),
                    1000 / float(max(ori_height, ori_width)))
        target_height, target_width = int(ori_height * scale), int(ori_width * scale)

        target_height = round2nearest_multiple(target_height, 8)
        target_width = round2nearest_multiple(target_width, 8)

        img_resized = cv2.resize(img.copy(), (target_width, target_height))

        img_resized = img_transform(img_resized)
        img_resized = torch.unsqueeze(img_resized, 0)
        img_resized_list.append(img_resized)

    batch_data = dict()
    batch_data['img_ori'] = img.copy()
    batch_data['img_data'] = [x.contiguous() for x in img_resized_list]
    segSize = (batch_data['img_ori'].shape[0],
                       batch_data['img_ori'].shape[1])
    img_resized_list = batch_data['img_data']

    scores = torch.zeros(1, 150, segSize[0], segSize[1])
    scores = async_copy_to(scores, 0)
    device = torch.device('cpu')
    scores = scores.to(device)
    for img in img_resized_list:
        feed_dict = batch_data.copy()
        feed_dict['img_data'] = img
        del feed_dict['img_ori']
        feed_dict = async_copy_to(feed_dict, 0)

        pred_tmp = segmentation_module(feed_dict, segSize=segSize)
        scores = scores + pred_tmp / len(imgSize)
    _, pred = torch.max(scores, dim=1)
    pred = as_numpy(pred.squeeze(0).cpu())

    mask = 1 - torch.from_numpy(
        np.logical_or(
            np.logical_or(
                np.logical_or(
                    np.logical_or(
                        np.logical_or(
                            np.logical_or(
                                np.logical_or(
                                    np.logical_or(
                                        np.logical_or(
                                            np.logical_or(pred == 12, pred == 20),
                                            pred == 76),
                                        pred == 80),
                                    pred == 83),
                                pred == 90),
                            pred == 102),
                        pred == 103),
                    pred == 116),
                pred == 126),
            pred == 127).astype(np.float32))
    return mask


































from scipy.signal import convolve2d
def movingstd2(A,k):
    A = A - torch.mean(A)
    Astd = torch.std(A)
    A = A/Astd
    A2 = A*A;
    
    wuns = torch.ones(A.shape)
    
    kernel = torch.ones(2*k+1,2*k+1)
    
    N =torch.nn.functional.conv2d(wuns.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0),padding=5)
    s = torch.sqrt((torch.nn.functional.conv2d(A2.unsqueeze(0).unsqueeze(0),kernel.unsqueeze(0).unsqueeze(0),padding=5) - ((torch.nn.functional.conv2d(A.unsqueeze(0).unsqueeze(0),kernel.unsqueeze(0).unsqueeze(0),padding=5))**2)/N)/(N-1))
    s = s*Astd
    
    return s

def moving_average(b, n=3) :
    res = gaussian_filter(b, sigma=n)
    return res

def grad_image(x):
    A = torch.Tensor([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    A = A.view((1,1,3,3))
    G_x = F.conv2d(x.unsqueeze(0).unsqueeze(0), A,padding=1)
    
    B = torch.Tensor([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
    B = B.view((1,1,3,3))
    G_y = F.conv2d(x.unsqueeze(0).unsqueeze(0), B,padding=1)
    
    G = torch.sqrt(torch.pow(G_x[0,0,:,:],2)+ torch.pow(G_y[0,0,:,:],2))
    return G


def detect_points(im1,im2):
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
   
    orb = cv2.ORB_create(2500)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
#    sift = cv2.xfeatures2d.SURF_create(2500)
#    keypoints1, descriptors1 = sift.detectAndCompute(im1Gray,None)
#    keypoints2, descriptors2 = sift.detectAndCompute(im2Gray,None)
    if len(keypoints1)==0 or len(keypoints2)==0:
        return None,None
   
    matcher = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
    matches = matcher.match(descriptors1, descriptors2, None)
   
    matches.sort(key=lambda x: x.distance, reverse=False)
 
    numGoodMatches = int(len(matches) * 0.15)
    matches = matches[:numGoodMatches]
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    return points1,points2


