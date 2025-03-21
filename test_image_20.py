import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import random

# ================== 配置参数 ==================
TEST_IMAGES_DIR = "/workspace/echo/image"  # 测试集目录路径
MODEL_PATH = "/workspace/echo/model/model_demo2.tflite"    # TFLite模型路径
LABELS = ["FireAxe", "FirstAidKit", "Flashlight", "Helmet"]                       # 替换为实际类别标签
INPUT_SIZE = (96, 96)                       # 模型输入尺寸
NORMALIZE_SCALE = 255.0                       # 归一化比例 (根据训练设置调整)
SAVE_NAME = "/workspace/echo/result/predictions_grid.png"            # 结果保存文件名

# ================== 1. 加载TFLite模型 ==================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# 获取模型输入/输出详细信息
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# ================== 2. 加载测试图片和真实标签 ==================
def load_random_images(dir_path, num_images=20):
    """从目录中加载图片及其真实标签"""
    all_paths = list(pathlib.Path(dir_path).glob("*/*.jpg"))  # 假设按子文件夹分类
    selected_paths = random.sample(all_paths, num_images)
    
    display_images = []    # 原始图片用于显示
    input_data = []        # 预处理后的数据用于推理
    true_labels = []       # 真实标签
    
    for path in selected_paths:
        # 从路径中提取真实标签（子文件夹名称）
        true_label = path.parent.name
        true_labels.append(true_label)
        
        # 加载并调整大小
        img = tf.keras.utils.load_img(path, target_size=INPUT_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        
        # 保存原始图片用于显示
        display_images.append(img_array.astype("uint8"))
        
        # 预处理 (根据训练时的预处理方式)
        processed_img = img_array / NORMALIZE_SCALE
        
        # 量化处理 (仅当输入类型为int8时)
        if input_details['dtype'] == np.int8:
            input_scale = input_details['quantization'][0]
            input_zero_point = input_details['quantization'][1]
            processed_img = processed_img / input_scale + input_zero_point
            processed_img = np.clip(processed_img, -128, 127).astype(np.int8)
        else:
            processed_img = processed_img.astype(input_details['dtype'])
        
        input_data.append(processed_img)
    
    return display_images, np.array(input_data), true_labels, selected_paths

display_images, input_data, true_labels, _ = load_random_images(TEST_IMAGES_DIR)

# ================== 3. 执行推理 ==================
predictions = []
for i in range(len(input_data)):
    interpreter.set_tensor(input_details['index'], input_data[i][np.newaxis, ...])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details['index'])[0]
    
    # 反量化处理 (仅当输出为int8时)
    if output_details['dtype'] == np.int8:
        output_scale = output_details['quantization'][0]
        output_zero_point = output_details['quantization'][1]
        output = (output.astype(np.float32) - output_zero_point) * output_scale
    
    predictions.append(output)

# 转换为预测结果
pred_indices = [np.argmax(p) for p in predictions]
pred_labels = [LABELS[i] for i in pred_indices]
confidences = [np.max(p) for p in predictions]

# ================== 4. 可视化结果 ==================
plt.figure(figsize=(12, 12))
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(display_images[i])
    
    # 获取预测和真实标签
    pred = pred_labels[i]
    true = true_labels[i]
    correct = (pred == true)
    
    # 构建标题文本
    title = [
        f"Pred: {pred} ({confidences[i]:.2f})",
        f"True: {true}",
        "✓" if correct else "✗"
    ]
    
    # 设置颜色（绿色正确，红色错误）
    color = "green" if correct else "red"
    plt.title("\n".join(title), color=color, fontsize=8)
    
    plt.axis('off')

plt.tight_layout()
plt.savefig(SAVE_NAME, dpi=150, bbox_inches='tight')
plt.show()