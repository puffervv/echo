# @Time    : 2024/3/31 13:27
# @Author  : 杨家威
# @File    : test_model.py
# @Software: VScode
# @Brief   : 测试tflite模型的程序，混淆矩阵会保存在result目录下
import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

# 获取当前时间戳
timestamp = str(int(time.time()))

# 加载TFLite模型
#model_path = './model/trained_Number.tflite'
model_path = '/workspace/echo/model/trained_Number_demo1.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 获取输入和输出张量
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 测试集目录
#test_dir = 'E:/car/material_tf/image_split/Number_split/test'
test_dir = '/workspace/echo/image_split/test'
class_names = sorted(os.listdir(test_dir))

# 创建result目录
result_dir = '/workspace/echo/result'
os.makedirs(result_dir, exist_ok=True)

# 初始化混淆矩阵
num_classes = len(class_names)
confusion_matrix = np.zeros((num_classes, num_classes))

# 遍历测试集,计算准确率和混淆矩阵
correct = 0
total = 0
confusion_matrix = np.zeros((num_classes, num_classes))
for class_name in class_names:
    class_dir = os.path.join(test_dir, class_name)
    for img_file in tqdm(os.listdir(class_dir),desc=f'img_file in {class_dir}'):
        img_path = os.path.join(class_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (96, 96))
        img = np.expand_dims(img, axis=0).astype(np.int8)

        # 获取预测结果
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data[0])
        actual_class_index = class_names.index(class_name)

        # 更新混淆矩阵和准确率
        confusion_matrix[actual_class_index, predicted_class] += 1
        if predicted_class == actual_class_index:
            correct += 1
        total += 1

# 计算测试准确率
accuracy = (correct / total) * 100

# 打印测试准确率
print(f'Test accuracy: {accuracy:.2f}%')

# 类别归一化混淆矩阵
normalized_confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=1, keepdims=True)

# 绘制并显示混淆矩阵的热力图
plt.figure(figsize=(12, 10))
plt.imshow(normalized_confusion_matrix, cmap='Blues', interpolation='nearest', aspect='auto')
plt.title('Normalized Confusion Matrix')
plt.colorbar()
plt.xticks(np.arange(num_classes), class_names, rotation=45)
plt.yticks(np.arange(num_classes), class_names)

# 在热力图上添加数值标签
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, f'{normalized_confusion_matrix[i, j]:.2f}', va='center', ha='center', fontsize=9)

plt.tight_layout()
plt.ylabel('Actual')
plt.xlabel('Predicted')
# 保存热力图到result文件夹
img_name = 'confusion_matrix'+timestamp+'.png'
result_img_path = os.path.join(result_dir, img_name)
plt.savefig(result_img_path)

# 显示热力图
plt.show()
