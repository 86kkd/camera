import tensorflow as tf

class Stand_Conv(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters= 32, kernel_size=(3,3), strides= (2,2), padding= 'valid')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU(max_value= 6)
    def call(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Depthwise_Conv(tf.keras.layers.Layer):
    def __init__(self, strides, padding):
        super().__init__()
        self.dw_conv = tf.keras.layers.DepthwiseConv2D(kernel_size= (3,3), strides= strides, padding= padding)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU(max_value= 6)
    def call(self,x):
        x = self.dw_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Pointwise_Conv(tf.keras.layers.Layer):
    def __init__(self, filters, alpha):
        super().__init__()
        self.pw_conv = tf.keras.layers.Conv2D(filters= int(filters * alpha), kernel_size= (1,1), strides= 1)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU(max_value= 6)
    def call(self,x):
        x = self.pw_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Depthwise_Separable_Conv(tf.keras.layers.Layer):
    def __init__( self,alpha, strides_depthwise, padding_depthwise, filters_pointwise):
        super().__init__()
        self.depth_wise = Depthwise_Conv(strides= strides_depthwise, padding= padding_depthwise)
        self.point_wise = Pointwise_Conv(filters= filters_pointwise, alpha= alpha)
    def call(self,x):
        x = self.depth_wise(x)
        x = self.point_wise(x)
        return x



class MobileNetV1(tf.keras.Model):
    def __init__(self,*, num_classes, depth_multiplyer = 1.0, droppout = 0.001,):
        super().__init__()
        assert depth_multiplyer > 0 and depth_multiplyer <= 1 ,'Error, my Mobilenet_V1 can only accept  alpha > 0 and alpha <= 1'

        self.std_conv = Stand_Conv()

        self.dp_conv1 = Depthwise_Separable_Conv(depth_multiplyer, strides_depthwise= (1,1), padding_depthwise= 'same',filters_pointwise= 64)
        self.dp_conv2 = Depthwise_Separable_Conv(depth_multiplyer, strides_depthwise= (2,2), padding_depthwise= 'valid', filters_pointwise= 128)
        self.dp_conv3 = Depthwise_Separable_Conv(depth_multiplyer, strides_depthwise= (1,1), padding_depthwise= 'same', filters_pointwise= 128)
        self.dp_conv4 = Depthwise_Separable_Conv(depth_multiplyer, strides_depthwise= (2,2), padding_depthwise= 'valid', filters_pointwise= 256)
        self.dp_conv5 = Depthwise_Separable_Conv(depth_multiplyer, strides_depthwise= (1,1), padding_depthwise= 'same', filters_pointwise= 256)
        self.dp_conv6 = Depthwise_Separable_Conv(depth_multiplyer, strides_depthwise= (2,2), padding_depthwise= 'valid', filters_pointwise= 512)
        self.dp_conv7 = Depthwise_Separable_Conv(depth_multiplyer, strides_depthwise= (1,1), padding_depthwise= 'same', filters_pointwise= 512)
        self.dp_conv8 = Depthwise_Separable_Conv(depth_multiplyer, strides_depthwise= (1,1), padding_depthwise= 'same', filters_pointwise= 512)
        self.dp_conv9 = Depthwise_Separable_Conv(depth_multiplyer, strides_depthwise= (1,1), padding_depthwise= 'same', filters_pointwise= 512)
        self.dp_conv10= Depthwise_Separable_Conv(depth_multiplyer, strides_depthwise= (1,1), padding_depthwise= 'same', filters_pointwise= 512)
        self.dp_conv11= Depthwise_Separable_Conv(depth_multiplyer, strides_depthwise= (1,1), padding_depthwise= 'same', filters_pointwise= 512)
        self.dp_conv12= Depthwise_Separable_Conv(depth_multiplyer, strides_depthwise= (2,2), padding_depthwise= 'valid', filters_pointwise= 1024)
        self.dp_conv13= Depthwise_Separable_Conv(depth_multiplyer, strides_depthwise= (2,2), padding_depthwise= 'same', filters_pointwise= 1024)

        self.gapool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(droppout)
        self.dense = tf.keras.layers.Dense(num_classes, activation= 'softmax')

    def call(self,x):
        x = self.std_conv(x)
        
        for i in range(1,14):
            layer = getattr(self, f'dp_conv{i}')
            x = layer(x)
        
        x = self.gapool(x)
        x = self.dropout(x)
        x = self.dense(x)
        # FC
        return x
        
    
if __name__ == '__main__':
    model = MobileNetV1(num_classes=15,depth_multiplyer=0.5)
    # model.build(input_shape=(None, 224, 224, 3))
    model.build(input_shape=(None,128,128,3))
    model.summary()
