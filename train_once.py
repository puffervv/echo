# @Time    : 2024/3/30 20:39
# @Author  : 杨家威
# @File    : train.py
# @Software: VScode
# @Brief   : mobilenet_v2模型训练代码，训练的模型会保存在models目录下，折线图会保存在result目录下
import tensorflow as tf
import numpy as np
import os
import time
from tensorflow import keras, lite
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

# 获取当前时间戳
timestamp = str(int(time.time()))

CLASS_NUM = 4
INPUT_SIZE = 96 #image size is 96 * 96 * 3
BITCH_SIZE = 128

train_datagen_augment = ImageDataGenerator(
                    rescale=1/255.0,
                    # rotation_range=20,        # 随机旋转的角度范围
                    # width_shift_range=0.2,     # 水平平移的范围（相对于总宽度的比例）
                    # height_shift_range=0.2,    # 垂直平移的范围（相对于总高度的比例）
                    shear_range=0.2,           # 剪切变换的范围
                    zoom_range=0.2,            # 随机缩放的范围
                    horizontal_flip=False,      # 随机水平翻转
                    # fill_mode='nearest'         # 填充新创建像素的方法)
                    )       

train_datagen_noaugment = ImageDataGenerator(rescale=1/255.0)

test_datagen = ImageDataGenerator(rescale=1/255.0)

training_set = train_datagen_augment.flow_from_directory('/workspace/echo/image_split/train',
                                                        target_size = (INPUT_SIZE, INPUT_SIZE),
                                                        batch_size = BITCH_SIZE,
                                                        class_mode = "categorical")

test_set = test_datagen.flow_from_directory('/workspace/echo/image_split/test',
                                            target_size = (INPUT_SIZE, INPUT_SIZE),
                                            batch_size = BITCH_SIZE,
                                            class_mode = "categorical")

base = keras.applications.MobileNetV2(
    include_top=False,
    alpha=0.5,
    weights="imagenet",
    input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
)
base.trainable = False

# x = base.output
# flatten = keras.layers.Flatten()(x)
# dropout = keras.layers.Dropout(0.3)(flatten)
# # x = tf.keras.layers.GlobalAveragePooling2D(name = "global_average_pooling_layer")(x)
# Dense = keras.layers.Dense(128, activation='relu')(dropout)
# dropout = keras.layers.Dropout(0.25)(Dense)
# predictions = keras.layers.Dense(CLASS_NUM,activation = 'softmax')(dropout)
# model = keras.models.Model(inputs=base.input, outputs=predictions)

# --------------------------
# 1. 输入层
# --------------------------
input_tensor = tf.keras.layers.Input(shape=(INPUT_SIZE,INPUT_SIZE, 3), name="input_layer")
# input_tensor = tf.keras.layers.experimental.preprocessing.Rescaling(1/255.)(input_tensor)

# --------------------------
# 2. 特征提取部分
# --------------------------  
x = base(input_tensor)
x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), name="avg_pool")(x)  # 明确池化尺寸
x = tf.keras.layers.Flatten(name="flatten")(x)
# x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(2, activation=None)(x)
# x = tf.keras.layers.Dropout(0.2)(x)

# --------------------------
# 3. 分类头
# --------------------------
outputs = tf.keras.layers.Dense(
    units=CLASS_NUM,
    activation='softmax', 
    kernel_initializer='he_normal',
    name="classification_head"
)(x)

# --------------------------
# 4. 构建完整模型
# --------------------------
model = tf.keras.models.Model(
    inputs=input_tensor,
    outputs=outputs,
    name="custom_classifier"
)


# tensorboard_callback = tf.keras.callbacks.TensorBoard(
#     log_dir='logs',  # 日志目录，TensorBoard将会在这里记录日志
#     histogram_freq=1,  # 记录直方图的频率
#     write_graph=True,  # 是否记录计算图
#     write_images=True    # 是否记录训练过程中的图片
# )

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                  patience=5)

base_learning_rate = 0.001  # 学习率
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

initial_epochs = 5 # 训练轮数

history = model.fit(training_set,
                    epochs=initial_epochs,
                    validation_data=test_set,
                    #callbacks=[tensorboard_callback],
                    callbacks=[early_stopping],
                    validation_freq=1,
                    verbose=1)

# loss0, accuracy0 = model.evaluate(test_set)
# print('Test accuracy0 :', accuracy0) # 无数据增强结果
# print('Test loss0 :', loss0)

# 学习曲线图
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
# plt.plot([initial_epochs-1,initial_epochs-1],
#           plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
# plt.plot([initial_epochs-1,initial_epochs-1],
#          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')

# 保存训练曲线图
result_dir = "/workspace/echo/result/"
result_img_name = 'training_map'+timestamp+'.png'
result_img_path = os.path.join(result_dir, result_img_name)
plt.savefig(result_img_path)

# 显示
plt.show()

# 最终准确率
loss, accuracy = model.evaluate(test_set)
print(f'Test Accuracy: {accuracy*100:.2f}%')  # 转换为百分比并保留两位小数
print(f'Test Loss: {loss:.4f}')

quant_set = test_datagen.flow_from_directory('/workspace/echo/image_split/test',
                        target_size = (INPUT_SIZE, INPUT_SIZE),
                        batch_size = 1,
                        class_mode = "categorical")

def representative_dataset():
    for i in range(100):
        x, y = quant_set.next()
        yield [x]

# Convert the tflite.
converter = lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_quant_model = converter.convert()

# Save the model.
#model.save("trained.h5")
model_dir = '/workspace/echo/model'
tflite_model_path = os.path.join(model_dir, 'trained_material_demo4.tflite')
with open(tflite_model_path, 'wb') as f:
  f.write(tflite_quant_model)