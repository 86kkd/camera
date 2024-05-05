import tensorflow as tf
import numpy as np
import cv2
# data argumentation

# def gamma_transform(tensor, gamma):
#     if tensor.shape.ndims == 3:
#         tensor = tf.expand_dims(tensor, axis=-1)
#     # 将BGR转换为HSV
#     hsv_tensor = tf.image.rgb_to_hsv(tensor)
#     illum = hsv_tensor[ ...,1:2]
#     illum = tf.pow(illum, gamma)
    
#     v_channel = tf.clip_by_value(illum ,  0., 1)

#     modified_hsv_tensor = tf.concat([
#         hsv_tensor[..., :1],
#         v_channel[...,:1],
#         hsv_tensor[...,-1:]
#     ], axis=-1)
#     tensor = tf.image.hsv_to_rgb(modified_hsv_tensor)

#     return tensor

def argument(x,y):
    x = tf.image.random_contrast(x, 0.8, 1.2)
    x = tf.image.random_saturation(x,0.8,1.2)

    # x = gamma_transform(x,2)
    
    return x,y

def normlize(x,y):

    # x = x/255
    # IMAGENET_DEFAULT_MEAN = tf.constant([0.485, 0.456, 0.406])
    # IMAGENET_DEFAULT_STD = tf.constant([0.229, 0.224, 0.225])
    # mean_array = tf.expand_dims(IMAGENET_DEFAULT_MEAN, axis=0)
    # std_array = tf.expand_dims(IMAGENET_DEFAULT_STD, axis=0)
    # x = (x - mean_array) / std_array

    return x,y

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse
    parser = argparse.ArgumentParser(description='input argumnet')
    parser.add_argument('--cls-file',default='class.txt',metavar='str',
                        help='class type file')
    parser.add_argument('--data-set',default='data_set/valiation',metavar='str',
                        help='where is data to train ')
    parser.add_argument('--image-size',default=224,metavar='int',
                    help="size of image to feed model")
    args = parser.parse_args()
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
        batch_size=9,
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
    val_data = val_data.map(argument)
    for i, (image, label) in enumerate(val_data):
        fig, axes = plt.subplots(3, 3)  # 创建3x3的子图
        for img, lab, ax in zip(image,label,axes.flatten()):
            ax.imshow(img.numpy().astype('uint32'))  # 显示图片
            ax.axis('off')  # 关闭坐标轴
            ax.set_title(f'lable:{lab}')
        plt.show()  # 显示所有的子图     
