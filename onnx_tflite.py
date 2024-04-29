import tensorflow as tf
import matplotlib.pyplot as plt
import os
import onnx
from onnx_tf.backend import prepare
from PIL import Image
import numpy as np
# from onnx_tflite import representative_dataset
from tqdm import tqdm
def representative_dataset():
    # 定义图像转换操作
    def load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])

        img /= 255.0
        IMAGENET_DEFAULT_MEAN = tf.constant([0.485, 0.456, 0.406])
        IMAGENET_DEFAULT_STD = tf.constant([0.229, 0.224, 0.225])
        mean_array = tf.expand_dims(IMAGENET_DEFAULT_MEAN, axis=0)
        std_array = tf.expand_dims(IMAGENET_DEFAULT_STD, axis=0)
        img = (img - mean_array) / std_array

        img = tf.expand_dims(img, 0)  # 添加一个批次维度
        # img = tf.transpose(img, [0, 3, 1, 2])  # 重排维度为 (batch, channels, height, width)
        # img = tf.cast(img, tf.float32)

        # print(tf.reduce_min(img))
        # print(tf.reduce_max(img))
        # print(img.shape)
        return img

    # 遍历数据集目录
    for root, _, files in os.walk("./data/val"):
        if not files:
            continue
        for file in tqdm(files, desc="\033[94mRepresentativeDataset\033[0m"):
            if file.lower().endswith(('.jpg', '.png')):
                # 构建完整的文件路径
                root_file = os.path.join(root, file)
                # 应用转换操作
                img = load_image(root_file)
                # 产生一个包含图像张量的字典
                yield {"input":img}

def onnx_forward(onnx_file, example_input):
    import onnxruntime

    sess_options = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(onnx_file, sess_options)
    input_name = session.get_inputs()[0].name
    example_input = tf.transpose(example_input, perm=[0, 3, 1, 2 ])
    output = session.run([], {input_name: example_input.numpy()})
    output = output[0]
    return output

# 指定数据集目录和模型路径
data_directory = 'data/val'
model_path = './mobilenetv3_small_025_val.onnx'

onnx_model = onnx.load(model_path)
tf_rep = prepare(onnx_model)
tf_rep.export_graph('/tmp/tf_model')

model = tf.saved_model.load("/tmp/tf_model")


# 定义一个简单的 tf.Module 用于保存模型
class SimpleModel(tf.Module):
    def __init__(self, original_model):
        self.original_model = original_model
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 224, 224,3], dtype=tf.float32)])
    def __call__(self, input):
        # 将输入张量的维度从 NCHW 转换为 NHWC
        input_tensor_nchw_to_nhwc = tf.transpose(input, perm=[0, 3, 1, 2 ])
        # 使用转换后的张量进行模型推理
        output = self.original_model.signatures['serving_default'](input0=input_tensor_nchw_to_nhwc)['output0']
        return output
    
# 创建模型实例
simple_model = SimpleModel(model)


# 构建数据集
def load_image(path):
    """加载一个图像并将其转换为张量"""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32)
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
    img_normalized = (img - mean_array) / std_array

    return img_normalized

# 使用 tf.data 从目录构建数据集
dataset = tf.data.Dataset.list_files(f'{data_directory}/*/*.jpg')  # 根据你的文件格式调整
dataset = dataset.map(load_image)
dataset = dataset.map(lambda img: (preprocess_image(img), img))
dataset = dataset.batch(1)  # 一次性处理 1 张图像

# 对数据集进行推理
for batch, (images, display_images) in enumerate(dataset):
    # 由于现在每次处理一张图片，所以不需要 take(1) 限制和 batch 条件判断
    predictions = simple_model(images)
    
    # 获取最可能的类别标签的索引
    predicted_label = tf.argmax(tf.squeeze(predictions), axis=0).numpy() # 使用 .numpy() 来获取 Python 整数

    onnx_out = onnx_forward(model_path, images)

    # 显示图像和对应的预测标签
    fig, axs = plt.subplots(1, 1, figsize=(20, 2))
    # display_images = tf.transpose(display_images, perm=[0, 3, 1, 2 ])
    img = (tf.squeeze(display_images).numpy()).astype('uint8')  # 转换回 uint8 类型
    axs.imshow(img)
    axs.set_title(f'Label: {predicted_label} onnx:{onnx_out.argmax()}')
    print(f'Predicted label: {predicted_label} onnx:{onnx_out.argmax()}')
    axs.axis('off')
    plt.show()
    if batch == 2:
        # 一旦达到预设的批次数量，就停止循环
        break

tf.saved_model.save(simple_model, '/tmp/mobilenetv3_val')
converter = tf.lite.TFLiteConverter.from_saved_model('/tmp/mobilenetv3_val')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
# converter._experimental_disable_per_channel = False
print(f"\033[94mConverting model\033[0m")
tflite_model = converter.convert()

import pathlib
tflite_models_dir = pathlib.Path("/tmp/tflite_models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model_file = tflite_models_dir/"mobilenetv3_val.tflite"
tflite_model_file.write_bytes(tflite_model)