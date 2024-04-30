import tensorflow as tf


batch_size = 50
def load_image(path):
    """加载一个图像并将其转换为张量"""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    print(f"=================={str(path)}===================")
    # img = tf.transpose(img, [2, 0, 1])  # 重排维度为 (batch, channels, height, width)
    return img

def preprocess_image(img, size=(224, 224)):
    """对图像进行预处理，例如调整大小和归一化"""
    img = tf.image.resize(img, size)
    img = tf.cast(img, tf.float32)

    # 归一化
    # img /= 255.0
    # IMAGENET_DEFAULT_MEAN = tf.constant([0.485, 0.456, 0.406])
    # IMAGENET_DEFAULT_STD = tf.constant([0.229, 0.224, 0.225])
    # mean_array = tf.expand_dims(IMAGENET_DEFAULT_MEAN, axis=0)
    # std_array = tf.expand_dims(IMAGENET_DEFAULT_STD, axis=0)
    # img = (img - mean_array) / std_array

    return img

# tf.debugging.set_log_device_placement(True)

data_directory = "./data/val"
# 使用 tf.data 从目录构建数据集
dataset = tf.data.Dataset.list_files(f'{data_directory}/*/*.jpg')  # 根据你的文件格式调整
dataset = dataset.map(load_image)  #string to image
# list(dataset.as_numpy_iterator())[0]
dataset = dataset.map(lambda img: (preprocess_image(img), img)) #lambda in:out
dataset = dataset.batch(batch_size)  # 一次性处理 1 张图像