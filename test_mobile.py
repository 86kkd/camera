from network.mobilenet_small import MobileNetV3Small
import tensorflow as tf
import os
from tqdm import tqdm
import pathlib

def representative_dataset():
    # 定义图像转换操作
    def load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])

        img /= 255.0
        IMAGENET_DEFAULT_MEAN = tf.constant([0.485, 0.456, 0.406])
        IMAGENET_DEFAULT_STD = tf.constant([0.229, 0.224, 0.225])
        mean_array = tf.expand_dims(IMAGENET_DEFAULT_MEAN, axis=0)
        std_array = tf.expand_dims(IMAGENET_DEFAULT_STD, axis=0)
        img = (img - mean_array) / std_array

        img = tf.expand_dims(img, 0)  # 添加一个批次维度\
        img = tf.cast(img,tf.int8)
        # img = tf.transpose(img, [0, 3, 1, 2])  # 重排维度为 (batch, channels, height, width)
        # img = tf.cast(img, tf.float32)

        # print(tf.reduce_min(img))
        # print(tf.reduce_max(img))
        # print(img.shape)
        return img

    # 遍历数据集目录
    for root, _, files in os.walk("./data/val/A-firearms-0"):
        if not files:
            continue
        for file in tqdm(files, desc="\033[94mRepresentativeDataset\033[0m"):
            if file.lower().endswith(('.jpg', '.png')):
                # 构建完整的文件路径
                root_file = os.path.join(root, file)
                # 应用转换操作
                img = load_image(root_file)
                # 产生一个包含图像张量的字典
                yield [tf.random.uniform(shape=(1, 224, 224, 3), dtype=tf.float32)]
                # yield {"input0":img}


if __name__ == '__main__':
    model = MobileNetV3Small()
    model.build(input_shape=(None, 224, 224, 3))
    # for layer in model.layers:
        # print(layer.input_shape)
    output = model(tf.zeros([1, 224, 224, 3]))
    
    quantilezed = True
    train = False or quantilezed

    if train :
        model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        model.fit(tf.zeros([1, 224, 224, 3]), 
                tf.zeros([1, 1]),  
                epochs=1) 

    model.summary()
    tflite_models_dir = pathlib.Path("/tmp/tf_model1")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    
    if quantilezed:
        
        tf.saved_model.save(model, tflite_models_dir)

        converter = tf.lite.TFLiteConverter.from_saved_model(tflite_models_dir._str)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.target_spec.supported_types = [tf.int8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
        # converter._experimental_disable_per_channel = False
        print(f"\033[94mConverting model\033[0m")
        tflite_model = converter.convert()

        import pathlib
        tflite_models_dir = pathlib.Path("output/mv3_tftest")
        tflite_models_dir.mkdir(exist_ok=True, parents=True)

        tflite_model_file = tflite_models_dir/"mobilenetv3_tf.tflite"
        tflite_model_file.write_bytes(tflite_model)