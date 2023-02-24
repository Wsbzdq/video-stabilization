### 数据预处理 ###
### 将一段视频作为输入，将其按帧分割所谓后续操作的输入 ###
import cv2
import os


def save_img(video_path):
    # 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
    videos = os.listdir(video_path)
    for video_name in videos:
        # 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则分隔 num+1 个子字符串
        file_name = video_name.split('.')[0]
        # 路径拼接
        folder_name = os.path.join(video_path, file_name)
        if not os.path.exists(folder_name):
            # 创建目录
            os.makedirs(folder_name, exist_ok=True)
        # 载入视频内容
        vc = cv2.VideoCapture(video_path+video_name)
        c = 0
        # 判断载入的视频是否可以打开
        rval = vc.isOpened()

        # 循环读取视频帧
        while rval:
            # 进行单张图片的读取，rval的值为True或者Flase， frame表示读入的图片
            rval, frame = vc.read()
            if rval:
                frames = c + 10000
                # 10000是啥？
                #cv2.imwrite（1."图片名字.格式"，2.Mat类型的图像数据，3.特定格式保存的参数编码，默认值std::vector < int > ()所以一般可以不写
                # 存储为图像,保存名为 文件夹名_数字（第几个文件）.jpg
                frame = cv2.resize(frame,(640,360))
                cv2.imwrite(os.path.join(folder_name, str(c) + '.jpg'), frame)
                cv2.waitKey(1)
                c = c + 1
            else:
                break
        vc.release()
        print('save_success')
        print(folder_name)
if __name__ == '__main__':
    # path to video folds eg: video_path = './Test/'
    #video_path = './Test/'
    #save_img(video_path)
    video_path = './Test/'
    save_img(video_path)
    