import tensorflow as tf
import matplotlib.pyplot as plt
import os
import onnx
from onnx_tf.backend import prepare
from PIL import Image
import numpy as np
# from onnx_tflite import representative_dataset
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="eval tflite model")
parser.add_argument('--lite',default='/tmp/tflite_model',metavar='str',
                    help='path to tflite model')
parser.add_argument('--val',default='data/valiation',metavar='str',
                    help='path to valiation data')
args = parser.parse_args()
# 构建数据集
def load_image(path):
    """加载一个图像并将其转换为张量"""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    # img = tf.transpose(img, [2, 0, 1])  # 重排维度为 (batch, channels, height, width)
    return img

def preprocess_image(img, size=(224, 224)):
    """对图像进行预处理，例如调整大小和归一化"""
    img = tf.image.resize(img, size)
    img = tf.cast(img, tf.float32)
    
    img /= 255.0
    IMAGENET_DEFAULT_MEAN = tf.constant([0.485, 0.456, 0.406])
    IMAGENET_DEFAULT_STD = tf.constant([0.229, 0.224, 0.225])
    mean_array = tf.expand_dims(IMAGENET_DEFAULT_MEAN, axis=0)
    std_array = tf.expand_dims(IMAGENET_DEFAULT_STD, axis=0)
    img = (img - mean_array) / std_array

    return img
data_directory = args.val
# 使用 tf.data 从目录构建数据集
dataset = tf.data.Dataset.list_files(f'{data_directory}/*/*.jpg')  # 根据你的文件格式调整
dataset = dataset.map(load_image)
dataset = dataset.map(lambda img: (preprocess_image(img), img))
dataset = dataset.batch(1)  # 一次性处理 1 张图像

interpreter = tf.lite.Interpreter(model_path=args.lite)
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.allocate_tensors()

print(input_details)
print(output_details)

# 对数据集进行推理
for batch, (images, display_images) in enumerate(dataset):
    # 由于现在每次处理一张图片，所以不需要 take(1) 限制和 batch 条件判断

    input_type = input_details[0]['dtype']
    if input_type == np.int8:
        input_scale, input_zero_point = input_details[0]['quantization']
        print("Input scale:", input_scale)
        print("Input zero point:", input_zero_point)
        images = (images / input_scale) + input_zero_point
        # images = (images / 1.0) - 127
        images = tf.cast(images,input_details[0]["dtype"])

    interpreter.set_tensor(input_details[0]['index'], images) 

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 获取最可能的类别标签的索引
    predicted_label = tf.argmax(tf.squeeze(output_data), axis=0).numpy() # 使用 .numpy() 来获取 Python 整数


    # 显示图像和对应的预测标签
    fig, axs = plt.subplots(1, 1, figsize=(20, 2))
    # display_images = tf.transpose(display_images, perm=[0, 3, 1, 2 ])
    img = (tf.squeeze(display_images).numpy()).astype('uint8')  # 转换回 uint8 类型
    axs.imshow(img)
    axs.set_title(f'Label: {predicted_label} ')
    print(f'Predicted label: {predicted_label}')
    axs.axis('off')
    plt.show()


