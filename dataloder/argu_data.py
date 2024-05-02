import argparse

parser = argparse.ArgumentParser(description='input argumnet')
parser.add_argument('--cls-file',default='class.txt',metavar='str',
                    help='class type file')

import tensorflow as tf
# data argumentation
def arugment(x,y):
    image = tf.image.random_brightness(x,max_delta=0.05)
    return image,y

def normlize(x,y):

    # x = x/255
    # IMAGENET_DEFAULT_MEAN = tf.constant([0.485, 0.456, 0.406])
    # IMAGENET_DEFAULT_STD = tf.constant([0.229, 0.224, 0.225])
    # mean_array = tf.expand_dims(IMAGENET_DEFAULT_MEAN, axis=0)
    # std_array = tf.expand_dims(IMAGENET_DEFAULT_STD, axis=0)
    # x = (x - mean_array) / std_array

    return x,y

if __name__ == '__main__':

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
