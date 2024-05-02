import argparse

parser = argparse.ArgumentParser(description="eval tflite model")
parser.add_argument('--lite',default='/tmp/tflite_model/out_put.tflite',metavar='str',
                    help='path to tflite model')
parser.add_argument('--val',default='data_set/valiation',metavar='str',
                    help='path to valiation data')
parser.add_argument('--image-size',default=224,metavar='int',
                    help="size of image to feed model")
parser.add_argument('--cls-file',default='class.txt',metavar='str',
                    help='class type file')

args = parser.parse_args()

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from dataloder.argu_data import normlize

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

print(input_details)
print(output_details)

# 对数据集进行推理
for batch, (images, lable) in enumerate(val_data):
    # 由于现在每次处理一张图片，所以不需要 take(1) 限制和 batch 条件判断
    display_image = images
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
    img = (tf.squeeze(display_image).numpy()).astype('uint8')  # 转换回 uint8 类型
    axs.imshow(img)
    axs.set_title(f'predicted: {predicted_label} label:{lable}')
    print(f'the shape of output_data:{output_data}')
    print(f'Predicted label: {predicted_label}')
    axs.axis('off')
    plt.show()


