import numpy as np
from tqdm import tqdm
import cv2
import random
import os
import time

# 指定路径
path = '/workspace/echo/archive'
# 指定新文件的路径
new_path = '/workspace/echo/image'
# 指定背景图路径
background_path = '/workspace/echo/174.png'

# 重复数据增强次数num
for num in range(1):
    # 获取当前时间戳
    timestamp = str(int(time.time()))
    # 遍历path目录下的每个文件夹
    for foldername in tqdm(os.listdir(path), desc='Processing folders'):
        folder_path = os.path.join(path, foldername)
        if os.path.isdir(folder_path):  # 确保是文件夹
            # 在new_path下创建与原文件夹同名的文件夹
            new_folder = os.path.join(new_path, foldername)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            
            # 遍历文件夹中的所有png文件
            for filename in tqdm(os.listdir(folder_path), desc=f'Processing images in {foldername}',leave=False,position=1):
                if filename.endswith('.png'):
                    # 获取文件完整路径
                    image_path = os.path.join(folder_path, filename)

                    # 读取图像
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"无法读取图像: {image_path}")
                        continue  # 跳过无法读取的图像
                    # 定义新的文件名和路径
                    base, extension = os.path.splitext(filename)  # 分离文件名和扩展名
                    new_filename = base + timestamp + extension  # 新文件名
                    new_image_path = os.path.join(new_folder, new_filename)
                    
                    # 修改分辨率并保存
                    image = cv2.resize(img, (224, 224))
                    # # 读取背景图
                    # background = cv2.imread(background_path)
                    # background = cv2.resize(background,(1000,1000))
                    # # 背景图尺寸
                    # background_height, background_width = background.shape[:2]
                    # # 贴图尺寸
                    # patch_height, patch_width = image.shape[:2]

                    # # 计算居中位置的坐标
                    # start_x = (background_width - patch_width) // 2
                    # start_y = (background_height - patch_height) // 2


                    # # 确保起始坐标不会超出背景图的边界
                    # start_x = max(0, start_x)
                    # start_y = max(0, start_y)

                    # # 计算粘贴区域的尺寸，确保不会超出背景图的边界
                    # end_x = min(start_x + patch_width, background_width)
                    # end_y = min(start_y + patch_height, background_height)

                    # background[start_y:end_y, start_x:end_x] = image[0:end_y-start_y, 0:end_x-start_x]


                    # 修改颜色空间为HSV
                    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    turn_hsv = img_hsv.copy()
                    # 调整HSV三通道，H-色调[0,180]，S-饱和度[0,255]，V-明暗度[0,255]
                    turn_hsv[:,:,0]=(turn_hsv[:,:,0]+random.uniform(0.0,1.0))
                    turn_hsv[:,:,1]=(turn_hsv[:,:,1]+random.uniform(0.0,1.0))
                    turn_hsv[:,:,2]=(turn_hsv[:,:,2]+random.uniform(0.0,1.0))
                    turn_img = cv2.cvtColor(turn_hsv, cv2.COLOR_HSV2BGR)

                    # gamma矫正，调整图像的亮度和对比度
                    gamma = random.uniform(0.5, 1.5)
                    inv_gamma = 1 / gamma
                    table = np.array([((i / 255) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
                    turn_img = cv2.LUT(turn_img, table)

                    # 随机旋转
                    width, height = turn_img.shape[:2]
                    theta = random.uniform(0, 360)
                    img_change = cv2.getRotationMatrix2D((width / 2, height / 2), theta, 1)
                    img = cv2.warpAffine(turn_img, img_change, (width, height))

                    # while(theta > 90):
                    #     theta = theta-90
                    # # 转弧度
                    # theta_hu =  np.radians(theta)
                    # # 计算外接矩形边长
                    # new_d = int(patch_height*np.cos(theta_hu)+patch_height*np.sin(theta_hu))
                    # # 计算新的坐标起始点
                    # start_x = (background_width - new_d) // 2
                    # start_y = (background_height - new_d) // 2
                    # # 截图背景中外接矩形大小的区域为新的图片
                    # background = img[start_x:start_x+new_d,start_y:start_y+new_d]
                    
                    # 保存旋转和调整亮度后的图片
                    cv2.imwrite(new_image_path, img)

