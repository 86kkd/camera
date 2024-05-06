import argparse

parser = argparse.ArgumentParser(description="eval tflite model")
parser.add_argument('--lite',default='output/mobilenetv1_520/tflite_model/out_put.tflite',metavar='str',
                    help='path to tflite model')
parser.add_argument('--val',default='data_set/2024_05_06_val/image_qvga_800',metavar='str',
                    help='path to valiation data')
parser.add_argument('--image-size',default=128,metavar='int',
                    help="size of image to feed model")
parser.add_argument('--cls-file',default='data_set/2024_05_06_val/image_qvga_800/class.txt',metavar='str',
                    help='class type file')
parser.add_argument('--visiable',action='store_true',
                    help='if visiable')
args = parser.parse_args()

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from dataloder.argu_data import normlize
from tqdm import tqdm

if args.cls_file:
    with open(args.cls_file, 'r') as file:
        class_names = file.read().splitlines()
else:
    class_names = None


val_data = tf.keras.preprocessing.image_dataset_from_directory(
    args.val,
    labels='inferred',
    label_mode='int',
    class_names=class_names,
    color_mode='rgb',
    batch_size=1,
    image_size=(args.image_size, args.image_size),
    shuffle=True,
    seed=1,
    # validation_split=1.0,
    # subset='validation',
    interpolation='bilinear',
    follow_links=True,
    crop_to_aspect_ratio=False,
    # pad_to_aspect_ratio=False,  # 填充（pad）到一个正方形
    # data_format=None,  # channel_first or chalnnel_last
    # verbose=True
)

val_data = val_data.map(normlize)

interpreter = tf.lite.Interpreter(model_path=args.lite)
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.allocate_tensors()

# print(input_details)
# print(output_details)

correct = 0
for batch, (images, lable) in tqdm(enumerate(val_data)):
    # 由于现在每次处理一张图片，所以不需要 take(1) 限制和 batch 条件判断
    display_image = images
    input_type = input_details[0]['dtype']
    if input_type == np.int8:
        input_scale, input_zero_point = input_details[0]['quantization']
        images = (images / input_scale) + input_zero_point
        # images = (images / 1.0) - 127
        images = tf.cast(images,input_details[0]["dtype"])

    interpreter.set_tensor(input_details[0]['index'], images) 
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 获取最可能的类别标签的索引
    predicted_label = tf.argmax(tf.squeeze(output_data), axis=0) # 使用 .numpy() 来获取 Python 整数

    if args.visiable:
        # 显示图像和对应的预测标签
        fig, axs = plt.subplots(1, 1, figsize=(20, 2))
        # display_images = tf.transpose(display_images, perm=[0, 3, 1, 2 ])
        img = (tf.squeeze(display_image).numpy()).astype('uint8')  # 转换回 uint8 类型
        axs.imshow(img)
        axs.set_title(f'predicted: {predicted_label.numpy()} label:{lable}')
        axs.axis('off')
        plt.show()
    
    predicted_label = tf.cast(predicted_label,tf.float64)
    lable = tf.cast(lable,tf.float64)
    batch = tf.cast(batch,tf.float64)
    correct += tf.reduce_sum(tf.cast(predicted_label == lable,tf.float64))
    accracy = correct/(batch+1)
    # print(f'the result of evaluate {accracy*100:.3f}%')
print(f'\033[94mthe result of evaluate {accracy*100:.3f}%\033[0m')