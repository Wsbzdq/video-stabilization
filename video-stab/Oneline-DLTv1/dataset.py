from torch.utils.data import Dataset
import  numpy as np
import cv2, torch
import os

from PIL import Image


def make_mesh(patch_w,patch_h):
    # 函数返回一个有终点和起点的固定步长的排列，输出0-patch_w的序列
    x_flat = np.arange(0,patch_w)
    # np.newaxis插入新的维度，相当于给一个数组外再加个括号
    x_flat = x_flat[np.newaxis,:]
    # ones()返回一个全1的n维数组，同样也有三个参数：
    # shape（用来指定返回数组的大小）
    # dtype（数组元素的类型）
    # order（是否以内存中的C或Fortran连续（行或列）顺序存储多维数据）
    # 后两个参数都是可选的，一般只需设定第一个参数。
    y_one = np.ones(patch_h)
    y_one = y_one[:,np.newaxis]
    # 矩阵相乘
    x_mesh = np.matmul(y_one , x_flat)

    y_flat = np.arange(0,patch_h)
    y_flat = y_flat[:,np.newaxis]
    x_one = np.ones(patch_w)
    x_one = x_one[np.newaxis,:]
    y_mesh = np.matmul(y_flat,x_one)
    return x_mesh,y_mesh

# 图像的标准化处理
class TrainDataset(Dataset):
    def __init__(self, data_path, exp_path, patch_w=560, patch_h=315, rho=16):

        # 规定数据（图片）的尺寸
        self.imgs = open(data_path, 'r').readlines()
        # np.reshape（a，b，c）在不改变数据内容的情况下，改变一个数组的格式，参数及返回值
        # a：数组--需要处理的数据
        # newshape：新的格式--整数或整数数组，如(2,3)表示2行3列，新的形状应该与原来的形状兼容，即行数和列数相乘后等于a中元素的数量
        # 均值
        self.mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
        # 标准差
        self.std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.WIDTH = 640
        self.HEIGHT = 360
        self.rho = rho
        self.x_mesh, self.y_mesh = make_mesh(self.patch_w, self.patch_h)
        self.train_path = os.path.join(exp_path, 'Data/Train/')

    # 在调用的时候使用
    def __getitem__(self, index):

        # 给图片编号
        value = self.imgs[index]
        img_names = value.split(' ')

        #############################
        mask_img1_name = img_names[0].split('/')
        mask_img2_num = img_names[1].split('/')
        mask_img2_name = mask_img2_num[1].split('\n')
        #print(mask_img2_num[0])
        #print(mask_img2_name[0])
        #print(mask_img2_name)
        #print(img_names[0])

        img_1 = cv2.imread(self.train_path + img_names[0])
        img1s = cv2.imread(self.train_path + mask_img1_name[0] + "/mask_" + mask_img1_name[1],cv2.IMREAD_GRAYSCALE)###################

        img1s = np.expand_dims(img1s, axis=0)  #######################################
        #img1s = np.expand_dims(img1s, axis=3)  #######################################
        #print(img1s.shape)

        # 调整图片尺寸
        height, width = img_1.shape[:2]
        # 若图片尺寸不一致将其进行缩放
        if height != self.HEIGHT or width != self.WIDTH:
            img_1 = cv2.resize(img_1, (self.WIDTH, self.HEIGHT))

        img_1 = (img_1 - self.mean_I) / self.std_I



        # numpy.mean(a, axis, dtype, out，keepdims)
        # mean()函数功能：求取均值
        # 经常操作的参数为axis，以m * n矩阵举例：
        # axis 不设置值，对 m*n 个数求均值，返回一个实数
        # axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
        # axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
        img_1 = np.mean(img_1, axis=2, keepdims=True)



        # np.transpose在没有参数的情况下表示矩阵转置（对2维即以上的数组才起作用）
        img_1 = np.transpose(img_1, [2, 0, 1])
        #print(img_1.shape)

        img_2 = cv2.imread(self.train_path + img_names[1][:-1])

        img2s = cv2.imread(self.train_path + mask_img2_num[0] + "/mask_" + mask_img2_name[0], cv2.IMREAD_GRAYSCALE)  ###################
        img2s = np.expand_dims(img2s, axis=0)########################################
        #img2s = np.expand_dims(img2s, axis=3)  #######################################

        height, width = img_2.shape[:2]
        if height != self.HEIGHT or width != self.WIDTH:
            img_2 = cv2.resize(img_2, (self.WIDTH, self.HEIGHT))

        img_2 = (img_2 - self.mean_I) / self.std_I
        img_2 = np.mean(img_2, axis=2, keepdims=True)

        img_2 = np.transpose(img_2, [2, 0, 1])

        # np.concatenate能够一次完成多个数组的拼接
        org_img = np.concatenate([img_1, img_2], axis=0)
        org_imgs = np.concatenate([img1s, img2s], axis=0)
        #print(org_imgs.shape)
        #print(org_img.shape)

        # 返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)。
        # 如果没有写参数high的值，则返回[0,low)的值。
        x = np.random.randint(self.rho, self.WIDTH - self.rho - self.patch_w)
        y = np.random.randint(self.rho, self.HEIGHT - self.rho - self.patch_h)

        input_tesnor = org_img[:, y: y + self.patch_h, x: x + self.patch_w]

        y_t_flat = np.reshape(self.y_mesh, (-1))
        x_t_flat = np.reshape(self.x_mesh, (-1))
        patch_indices = (y_t_flat + y) * self.WIDTH + (x_t_flat + x)

        top_left_point = (x, y)
        bottom_left_point = (x, y + self.patch_h)
        bottom_right_point = (self.patch_w + x, self.patch_h + y)
        top_right_point = (x + self.patch_w, y)
        h4p = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]

        h4p = np.reshape(h4p, (-1))

        org_img = torch.tensor(org_img)
        org_imgs = torch.tensor(org_imgs)#########################
        #print(org_imgs.shape)
        input_tesnor = torch.tensor(input_tesnor)
        patch_indices = torch.tensor(patch_indices)
        h4p = torch.tensor(h4p)

        return (org_img, input_tesnor, patch_indices, h4p, org_imgs)

    def __len__(self):

        return len(self.imgs)


class TestDataset(Dataset):
    def __init__(self, data_path, patch_w=560, patch_h=315, rho=16, WIDTH=640, HEIGHT=360):
        self.mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
        self.std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.rho = rho
        self.x_mesh, self.y_mesh = make_mesh(self.patch_w,self.patch_h)

        self.work_dir = os.path.join(data_path, 'Data','Data')
        self.pair_list = list(open(os.path.join(self.work_dir, 'Test/test.txt')))
        print(len(self.pair_list))
        self.img_path = os.path.join(self.work_dir, 'Test/')
        self.npy_path = os.path.join(self.work_dir, 'Coordinate/')

    def __getitem__(self, index):

        img_pair = self.pair_list[index]
        pari_id = img_pair.split(' ')
        npy_id = pari_id[0].split('/')[1] + '_' + pari_id[1].split('/')[1][:-1] + '.npy'
        npy_id = self.npy_path + npy_id
        video_name = img_pair.split('/')[0]

        # load img1
        if pari_id[0][-1] == 'M':
            img_1 = cv2.imread(self.img_path + pari_id[0][:-2])
        else:
            img_1 = cv2.imread(self.img_path + pari_id[0])

        # load img2
        if pari_id[1][-2] == 'M':
            img_2 = cv2.imread(self.img_path + pari_id[1][:-3])
        else:
            img_2 = cv2.imread(self.img_path + pari_id[1][:-1])
        
        height, width = img_1.shape[:2]
 
        if height != self.HEIGHT or width != self.WIDTH:
            img_1 = cv2.resize(img_1, (self.WIDTH, self.HEIGHT))

        print_img_1 = img_1.copy()
        print_img_1 = np.transpose(print_img_1, [2, 0, 1])

        img_1 = (img_1 - self.mean_I) / self.std_I
        img_1 = np.mean(img_1, axis=2, keepdims=True)
        img_1 = np.transpose(img_1, [2, 0, 1])

        height, width = img_2.shape[:2]

        if height != self.HEIGHT or width != self.WIDTH:
            img_2 = cv2.resize(img_2, (self.WIDTH, self.HEIGHT))

        print_img_2 = img_2.copy()
        print_img_2 = np.transpose(print_img_2, [2, 0, 1])
        img_2 = (img_2 - self.mean_I) / self.std_I
        img_2 = np.mean(img_2, axis=2, keepdims=True)
        img_2 = np.transpose(img_2, [2, 0, 1])
        org_img = np.concatenate([img_1, img_2], axis=0)
        WIDTH = org_img.shape[2]
        HEIGHT = org_img.shape[1]

        x = np.random.randint(self.rho, WIDTH - self.rho - self.patch_w)
        x = 40  # patch should in the middle of full img when testing补丁应该在完整的img时进行测试
        y = np.random.randint(self.rho, HEIGHT - self.rho - self.patch_h)
        y = 23  # patch should in the middle of full img when testing补丁应该在完整的img时进行测试
        input_tesnor = org_img[:, y: y + self.patch_h, x: x + self.patch_w]

        y_t_flat = np.reshape(self.y_mesh, [-1])
        x_t_flat = np.reshape(self.x_mesh, [-1])
        patch_indices = (y_t_flat + y) * WIDTH + (x_t_flat + x)

        top_left_point = (x, y)
        bottom_left_point = (x, y + self.patch_h)
        bottom_right_point = (self.patch_w + x, self.patch_h + y)
        top_right_point = (x + self.patch_w, y)
        four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]

        four_points = np.reshape(four_points, (-1))

        return (org_img, input_tesnor, patch_indices, four_points,print_img_1, print_img_2, video_name, npy_id)

    def __len__(self):

        return len(self.pair_list)
