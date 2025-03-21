import tensorflow as tf
from tensorflow import keras, lite
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model('/workspace/echo/model/model.h5')

test_datagen = ImageDataGenerator(rescale=1/255.0)
quant_set = test_datagen.flow_from_directory('/workspace/echo/image_split/test',
                        target_size = (112, 112),
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

model_dir = '/workspace/echo/model'
tflite_model_path = os.path.join(model_dir, 'model_demo3.tflite')
with open(tflite_model_path, 'wb') as f:
  f.write(tflite_quant_model)

# 加载TFLite模型并分配张量
interpreter = tf.lite.Interpreter(model_path='/workspace/echo/model/model_demo3.tflite')
interpreter.allocate_tensors()

# 获取输入输出详细信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 准备测试输入
input_data = np.array(np.random.random_sample(input_details[0]['shape']), dtype=np.int8)
interpreter.set_tensor(input_details[0]['index'], input_data)

# 运行推理
interpreter.invoke()

# 获取输出结果
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)