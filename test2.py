import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 配置参数
TEST_DATA_DIR = '/workspace/echo/image_split/test'
INPUT_SIZE = 96  # 与模型输入尺寸一致
CLASS_NAMES = ["FireAxe", "FirstAidKit", "Flashlight", "Helmet"]  # 替换为实际类别名称

# 1. 准备测试数据
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255  # 保持与训练相同的预处理
)

test_generator = test_datagen.flow_from_directory(
    TEST_DATA_DIR,
    target_size=(INPUT_SIZE, INPUT_SIZE),
    batch_size=1,
    class_mode='categorical',
    shuffle=False  # 必须关闭shuffle以保证顺序
)

# 2. 加载量化模型
interpreter = tf.lite.Interpreter(model_path='/workspace/echo/model/trained_material_demo3.tflite')
interpreter.allocate_tensors()

# 获取模型IO配置
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. 进行预测
y_true = []
y_pred = []

for _ in range(len(test_generator)):
    # 获取数据
    x, y = next(test_generator)
    y_true.append(np.argmax(y))
    
    # 量化预处理（关键步骤）
    input_data = x.astype(input_details[0]['dtype'])
    
    # 设置输入张量
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # 执行推理
    interpreter.invoke()
    
    # 获取输出
    output = interpreter.get_tensor(output_details[0]['index'])
    y_pred.append(np.argmax(output))

# 4. 计算指标
accuracy = np.mean(np.array(y_true) == np.array(y_pred))
print(f"\nTest Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, zero_division = 1, target_names=CLASS_NAMES))

# 5. 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(y_true, y_pred, CLASS_NAMES)
