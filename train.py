import tensorflow as tf
from network.mobilenet_small import MobileNetV3Small
import argparse
from tqdm import tqdm
import datetime

parser = argparse.ArgumentParser(description="tf model train")
parser.add_argument('--batch-size','-b',default=511,metavar="N",
                    help="input batchsize for training")
parser.add_argument('--epochs',default=599,metavar="N",
                    help="input num train epochs")
parser.add_argument('--data-set',default='data',metavar='str',
                    help='where is data to train ,subval wirh train and val')
parser.add_argument('--image-size',default=224,metavar='N',
                    help="size of image to feed model")
parser.add_argument('--td',default='/tmp/td',metavar='str',
                    help='tensorboard save dir')
args = parser.parse_args()



model = MobileNetV3Small()
model.build(input_shape=(None, 224, 224, 3))
model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
                    
log_dir = args.td + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

gpus = tf.config.list_physical_devices('GPU')


train_data = tf.keras.preprocessing.image_dataset_from_directory(
    args.data_set,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=args.batch_size,
    image_size=(args.image_size, args.image_size),
    shuffle=True,
    seed=1,
    validation_split=0.1,
    subset='training',
    interpolation='bilinear',
    follow_links=True,
    crop_to_aspect_ratio=False,
    # pad_to_aspect_ratio=False,
    # data_format=None,
    # verbose=True
)

# with tf.device(gpus):
model.fit(train_data,
    epochs = args.epochs,
    steps_per_epoch = 18,
    callbacks = [tensorboard_callback])