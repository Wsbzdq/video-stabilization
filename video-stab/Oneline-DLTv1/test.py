# coding: utf-8
import argparse

import cv2
import cv2.cv2
import numpy
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from torch_homography_model import build_model
from dataset import *
from utils import transformer as trans
import os
import numpy as np
import time


def test(args):

    exp_name = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
    work_dir = os.path.join(exp_name, 'Data','Data')
    pair_list = list(open(os.path.join(work_dir, 'Test/test.txt')))
    npy_path = os.path.join(work_dir, 'Coordinate/')
    result_name = "exp_result_Oneline-FastDLT"
    result_files = os.path.join(exp_name, result_name)
    if not os.path.exists(result_files):
        os.makedirs(result_files)

    result_txt = "result_ours_exp.txt"
    res_txt = os.path.join(result_files, result_txt)
    f = open(res_txt, "w")

    net = build_model(args.model_name, pretrained=args.pretrained)
    if args.finetune == True:
        model_path = os.path.join(exp_name, 'resnet34_iter.pth')
        print(model_path)
        state_dict = torch.load(model_path, map_location='cpu')

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.state_dict().items():
            namekey = k[7:]
            new_state_dict[namekey] = v

        net = build_model(args.model_name)
        model_dict = net.state_dict()
        new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict.keys()}
        model_dict.update(new_state_dict)
        net.load_state_dict(model_dict)

    net = torch.nn.DataParallel(net)
    if torch.cuda.is_available():
        net = net.cuda()

    M_tensor = torch.tensor([[args.img_w/ 2.0, 0., args.img_w/ 2.0],
                             [0., args.img_h / 2.0, args.img_h / 2.0],
                             [0., 0., 1.]])
    if torch.cuda.is_available():
        M_tensor = M_tensor.cuda()

    M_tile = M_tensor.unsqueeze(0).expand(1, M_tensor.shape[-2], M_tensor.shape[-1])
    M_tensor_inv = torch.inverse(M_tensor)
    M_tile_inv = M_tensor_inv.unsqueeze(0).expand(1, M_tensor_inv.shape[-2], M_tensor_inv.shape[-1])

    test_data = TestDataset(data_path=exp_name, patch_w=args.patch_size_w, patch_h=args.patch_size_h, rho=16, WIDTH=args.img_w, HEIGHT=args.img_h)
    test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=0, shuffle=False, drop_last=True)

    print("start testing")
    net.eval()


    for i, batch_value in enumerate(test_loader):

        img_pair = pair_list[i]
        pari_id = img_pair.split(' ')
        npy_name = pari_id[0].split('/')[1] + '_' + pari_id[1].split('/')[1][:-1] + '.npy'
        npy_id = npy_path + npy_name


        org_imges = batch_value[0].float()
        input_tesnors = batch_value[1].float()
        patch_indices = batch_value[2].float()
        h4p = batch_value[3].float()
        print_img_1 = batch_value[4]
        print_img_2 = batch_value[5]
        org_imgs = batch_value[4].float()

        print_img_1_d = print_img_1.cpu().detach().numpy()[0, ...]
        print_img_2_d = print_img_2.cpu().detach().numpy()[0, ...]
        print_img_1_d = np.transpose(print_img_1_d, [1, 2, 0])
        print_img_2_d = np.transpose(print_img_2_d, [1, 2, 0])

        if torch.cuda.is_available():
            input_tesnors = input_tesnors.cuda()
            patch_indices = patch_indices.cuda()
            h4p = h4p.cuda()
            print_img_1 = print_img_1.cuda()

        batch_out = net(org_imges, input_tesnors, h4p, patch_indices, org_imgs)
        H_mat = batch_out['H_mat']

        output_size = (args.img_h, args.img_w)
   
        H_point = H_mat.squeeze(0)
        H_point = H_point.cpu().detach().numpy()
        H_point = np.linalg.inv(H_point)
        H_point = (1.0 / H_point.item(8)) * H_point


        name = "0"*(8-len(str(i)))+str(i)

        H_mat = torch.matmul(torch.matmul(M_tile_inv, H_mat), M_tile)


        if i == 0:
            homo = H_mat
            pred_full, _ = trans(print_img_1, homo, output_size)  # pred_full = warped imgA
        else:
            pred_full, _ = trans(print_img_1, homo, output_size)  # pred_full = warped imgA
            pred_full, _ = trans(print_img_1, H_mat, output_size)  # pred_full = warped imgA
            homo = torch.matmul(homo, H_mat)
        #pred_full, _ = trans(print_img_1, H_mat, output_size)

        pred_full = pred_full.cpu().detach().numpy()[0, ...]
        pred_full = pred_full.astype(np.uint8)

        input_list = [print_img_1_d, print_img_2_d]
        output_list = [pred_full, print_img_2_d]
        img_pair = img_pair.strip()
        target = img_pair.rsplit("_", 1)[1].split('.')[0]
        object = img_pair.split(".jpg")[0].split("_")[1]
        directory = img_pair.split("/")[0]+"/"+target
        print(directory)

        save_path = "../Data/Data/Test/10001"
        print(save_path + "/" + "10001_" + object + ".jpg")
        cv2.imwrite(save_path + "/"  + "10001" + "/" + "10001_" + object + ".jpg", pred_full)



    return 0


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=4, help='Number of splits')
    parser.add_argument('--cpus', type=int, default=10, help='Number of cpus')

    parser.add_argument('--img_w', type=int, default=640)
    parser.add_argument('--img_h', type=int, default=360)


    parser.add_argument('--patch_size_h', type=int, default=315)
    parser.add_argument('--patch_size_w', type=int, default=560)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-9, help='learning rate')

    parser.add_argument('--model_name', type=str, default='resnet34')
    parser.add_argument('--pretrained', type=bool, default=False, help='Use pretrained waights?')
    parser.add_argument('--finetune', type=bool, default=True, help='Use pretrained waights?')

    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)