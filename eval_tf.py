import argparse

parser = argparse.ArgumentParser(description="tf model train")
parser.add_argument('--batch-size','-b',default=1,metavar="int",
                    help="input batchsize for training")

parser.add_argument('--data-set',default='data_set/training',metavar='str',
                    help='where is data to val')
parser.add_argument('--image-size',default=224,metavar='int',
                    help="size of image to feed model")
parser.add_argument('--td',default='/tmp/td',metavar='str',
                    help='tensorboard save dir')
parser.add_argument('--model-path',default='/tmp/tf_model',metavar='str',
                    help='path to save model')

parser.add_argument('--cls-file',default='class.txt',metavar='str',
                    help='class type file')

args = parser.parse_args()

import tensorflow as tf
import matplotlib.pyplot as plt
from dataloder.argu_data import normlize

if args.cls_file:
    with open(args.cls_file, 'r') as file:
        class_names = file.read().splitlines()
else:
    class_names = None


val_data = tf.keras.preprocessing.image_dataset_from_directory(
    args.data_set,
    labels='inferred',
    label_mode='int',
    class_names=class_names,
    color_mode='rgb',
    batch_size=args.batch_size,
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

model = tf.saved_model.load(args.model_path)

for batch, (images, lable) in enumerate(val_data):
    # 由于现在每次处理一张图片，所以不需要 take(1) 限制和 batch 条件判断
    predictions = model(images)
    
    # 获取最可能的类别标签的索引
    predicted_label = tf.argmax(tf.squeeze(predictions), axis=0).numpy() # 使用 .numpy() 来获取 Python 整数

    # 显示图像和对应的预测标签
    fig, axs = plt.subplots(1, 1, figsize=(20, 2))
    # display_images = tf.transpose(display_images, perm=[0, 3, 1, 2 ])
    img = (tf.squeeze(images).numpy()).astype('uint8')  # 转换回 uint8 类型
    axs.imshow(img)
    axs.set_title(f'Label: {predicted_label} real label{lable}')
    print(f'Predicted label: {predicted_label} ')
    axs.axis('off')
    plt.show()