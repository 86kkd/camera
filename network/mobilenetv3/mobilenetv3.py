import tensorflow as tf
from network.mobilenetv3.mobilenetv3_black import BottleNeck, h_swish


class MobileNetV3(tf.keras.Model):
    def __init__(self,
                 num_classes=15,
                 depth_multiplyer=0.5,):
        super(MobileNetV3, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=16*depth_multiplyer,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            input_shape = (1,224,224,3),
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bneck1 = BottleNeck(in_size=16, exp_size=16, out_size=16, s=2, is_se_existing=True, NL="RE", k=3,  depth_multiplyer = depth_multiplyer)
        self.bneck2 = BottleNeck(in_size=16, exp_size=72, out_size=24, s=2, is_se_existing=False, NL="RE", k=3, depth_multiplyer = depth_multiplyer)
        self.bneck3 = BottleNeck(in_size=24, exp_size=88, out_size=24, s=1, is_se_existing=False, NL="RE", k=3, depth_multiplyer = depth_multiplyer)
        self.bneck4 = BottleNeck(in_size=24, exp_size=96, out_size=40, s=2, is_se_existing=True, NL="HS",  k=5, depth_multiplyer = depth_multiplyer)
        self.bneck5 = BottleNeck(in_size=40, exp_size=240, out_size=40, s=1, is_se_existing=True, NL="HS", k=5, depth_multiplyer = depth_multiplyer)
        self.bneck6 = BottleNeck(in_size=40, exp_size=240, out_size=40, s=1, is_se_existing=True, NL="HS", k=5, depth_multiplyer = depth_multiplyer)
        self.bneck7 = BottleNeck(in_size=40, exp_size=120, out_size=48, s=1, is_se_existing=True, NL="HS", k=5, depth_multiplyer = depth_multiplyer)
        self.bneck8 = BottleNeck(in_size=48, exp_size=144, out_size=48, s=1, is_se_existing=True, NL="HS", k=5, depth_multiplyer = depth_multiplyer)
        self.bneck9 = BottleNeck(in_size=48, exp_size=288, out_size=96, s=2, is_se_existing=True, NL="HS", k=5, depth_multiplyer = depth_multiplyer)
        self.bneck10 = BottleNeck(in_size=96, exp_size=576, out_size=96, s=1, is_se_existing=True, NL="HS",k=5, depth_multiplyer = depth_multiplyer)
        self.bneck11 = BottleNeck(in_size=96, exp_size=576, out_size=96, s=1, is_se_existing=True, NL="HS",k=5, depth_multiplyer = depth_multiplyer)

        self.conv2 = tf.keras.layers.Conv2D(filters=576*depth_multiplyer,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(7, 7),
                                                        strides=1)
        self.conv3 = tf.keras.layers.Conv2D(filters=1280,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.conv4 = tf.keras.layers.Conv2D(filters=num_classes,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)

        x = self.bn1(x, training=training)
        x = h_swish(x)
    
        x = self.bneck1(x, training=training)
        x = self.bneck2(x, training=training)
        x = self.bneck3(x, training=training)
        x = self.bneck4(x, training=training)
        x = self.bneck5(x, training=training)
        x = self.bneck6(x, training=training)
        x = self.bneck7(x, training=training)
        x = self.bneck8(x, training=training)
        x = self.bneck9(x, training=training)
        x = self.bneck10(x, training=training)
        x = self.bneck11(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = h_swish(x)
        
        x = self.avgpool(x)
        x = self.conv3(x)
        x = h_swish(x)
        x = self.conv4(x)

        return x


if __name__ == '__main__':
    model = MobileNetV3()
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()
