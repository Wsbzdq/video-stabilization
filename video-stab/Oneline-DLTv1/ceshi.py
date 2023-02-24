import cv2
from main import compute_mask
import os
import imageio

train_path1 = "../Data/Train/"
file1 = os.listdir(train_path1)
for imgs in file1:
    train_path2 = os.path.join(train_path1, imgs)
    file2 = os.listdir(train_path2)
    for img in file2:
        image = cv2.imread(train_path2+"/"+img)
        mask = compute_mask(image)
        imageio.imwrite(os.path.join(train_path2,'mask_'+img), mask)
        print(img)
    print(imgs)
print('save_success')

