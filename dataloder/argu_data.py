import tensorflow as tf
import numpy as np
import cv2
# data argumentation

def gamma_transform(img, gamma):
    is_gray = img.ndim == 2 or img.shape[1] == 1
    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    illum = hsv[..., 2] / 255.
    illum = np.power(illum, gamma)
    v = illum * 255.
    v[v > 255] = 255
    v[v < 0] = 0
    hsv[..., 2] = v.astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def argument(x,y):
    x = tf.image.random_contrast(x, 0.8, 1.2)
    x = tf.image.random_saturation(x,0.8,1.2)
    x_np = x.numpy()
    x = gamma_transform(x,2)
    # x = c2.convertScaleAbs(x.numpy(),alpha=1,beta=np.random.uniform(2.5, 5.0, size=1))
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
