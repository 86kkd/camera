import tensorflow as tf
from network.mobilenet_small import MobileNetV3Small
import argparse
from tqdm import tqdm
import datetime
import pathlib

parser = argparse.ArgumentParser(description="tf model train")
parser.add_argument('--batch-size','-b',default=800,metavar="N",
                    help="input batchsize for training")
parser.add_argument('--epochs',default=599,metavar="N",
                    help="input num train epochs")
parser.add_argument('--data-set',default='data/training',metavar='str',
                    help='where is data to train ,subval wirh train and val')
parser.add_argument('--image-size',default=224,metavar='N',
                    help="size of image to feed model")
parser.add_argument('--td',default='/tmp/td',metavar='str',
                    help='tensorboard save dir')
parser.add_argument('--save-path',default='/tmp/tf_model',metavar='str',
                    help='path to save model')
parser.add_argument('--num-cls',default=15,metavar='N',
                    help='num classes to classify')
parser.add_argument('--dm',default=0.5,metavar='float',
                    help='depth_multiplyer, change it to config total model parameter size')
parser.add_argument('--cls-file',default='class.txt',metavar='str',
                    help='class type file')
parser.add_argument('--val-rate',default=0.1,metavar='float',
                    help='rate to split input data into train and val data')
args = parser.parse_args()



model = MobileNetV3Small(num_classes=args.num_cls,depth_multiplyer=args.dm)
model.build(input_shape=(None, 224, 224, 3))
model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),
                             tf.keras.metrics.SparseTopKCategoricalAccuracy(
                                k=5, name='sparse_top_5_categorical_accuracy', dtype=None),
                             tf.keras.metrics.SparseTopKCategoricalAccuracy(
                                k=1, name='sparse_top_1_categorical_accuracy', dtype=None)
                            ]
)
model.summary()
log_dir = pathlib.Path(args.td)
log_dir = log_dir / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=str(log_dir), 
    histogram_freq=1,
    write_graph=True,
    # write_images=True,
    )

gpus = tf.config.list_physical_devices('GPU')

if args.cls_file:
    with open(args.cls_file, 'r') as file:
        class_names = file.read().splitlines()
else:
    class_names = None

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    args.data_set,
    labels='inferred',
    label_mode='int',
    class_names=class_names,
    color_mode='rgb',
    batch_size=args.batch_size,
    image_size=(args.image_size, args.image_size),
    shuffle=True,
    seed=1,
    validation_split=args.val_rate,
    subset='training',
    interpolation='bilinear',
    follow_links=True,
    crop_to_aspect_ratio=False,
    # pad_to_aspect_ratio=False,  # 填充（pad）到一个正方形
    # data_format=None,  # channel_first or chalnnel_last
    # verbose=True
)

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
    validation_split=args.val_rate,
    subset='validation',
    interpolation='bilinear',
    follow_links=True,
    crop_to_aspect_ratio=False,
    # pad_to_aspect_ratio=False,  # 填充（pad）到一个正方形
    # data_format=None,  # channel_first or chalnnel_last
    # verbose=True
)

# data argumentation
def augment(x,y):
    image = tf.image.random_brightness(x,max_delta=0.05)
    return image,y
train_data = train_data.map(augment)
# with tf.device(gpus):
try:
    model.fit(train_data,
        validation_data = val_data,
        epochs = args.epochs,
        steps_per_epoch = len(train_data),
        callbacks = [tensorboard_callback])
except KeyboardInterrupt:
    pass
tflite_models_dir = pathlib.Path(args.save_path)
tflite_models_dir.mkdir(exist_ok=True, parents=True)
print(f"\033[92m\n\nSavimg model to {tflite_models_dir._str}\033[0m\n")
tf.saved_model.save(model, tflite_models_dir)