import argparse

parser = argparse.ArgumentParser(description="tf model train")
parser.add_argument('--batch-size','-b',default=1,metavar="int",
                    help="input batchsize for training")

parser.add_argument('--data-set',default='data_set/val_bright/image_qvga_200',metavar='str',
                    help='where is data to val')
parser.add_argument('--image-size',default=224,metavar='int',
                    help="size of image to feed model")
parser.add_argument('--td',default='/tmp/td',metavar='str',
                    help='tensorboard save dir')
parser.add_argument('--model-path',default='output/mobilenetv1/best_model',metavar='str',
                    help='path to save model')

parser.add_argument('--cls-file',default='data_set/val_bright/image_qvga_200/class.txt',metavar='str',
                    help='class type file')

parser.add_argument('--visible',action='store_true',
                    help='if visual predict resulet')
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
    predicted_label = tf.argmax(tf.squeeze(predictions), axis=0)

    if args.visible:
        # 显示图像和对应的预测标签
        fig, axs = plt.subplots(1, 1, figsize=(20, 2))
        # display_images = tf.transpose(display_images, perm=[0, 3, 1, 2 ])
        img = (images[0].numpy()).astype('uint8')  # 转换回 uint8 类型
        axs.imshow(img)
        if args.batch_size!=1:
            axs.set_title(f'Label: {predicted_label[0].numpy()} real label{lable[0].numpy()}')
            print(f'Predicted label: {predicted_label[0].numpy()} ')
        else:
            axs.set_title(f'Label: {predicted_label.numpy()} real label{lable[0].numpy()}')
            print(f'Predicted label: {predicted_label.numpy()} ')
        axs.axis('off')
        plt.show()
    predicted_label = tf.cast(predicted_label,tf.float64)
    lable = tf.cast(lable,tf.float64)
    batch = tf.cast(batch,tf.float64)
    accracy = tf.reduce_sum(tf.cast(predicted_label == lable,tf.float64))/batch


print(f'the result of evaluate {accracy}')