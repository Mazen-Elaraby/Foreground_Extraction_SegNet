from keras.models import *
from keras.layers import *

def get_vgg16_encoder(ip_height,  ip_width, channels):
    
    #Block: 1
    img_ip = Input(shape=(ip_height, ip_width, channels))
    E = Conv2D(64, (3, 3), activation='relu', padding='same',name='block1_conv1')(img_ip)
    E = Conv2D(64, (3, 3), activation='relu', padding='same',name='block1_conv2')(E)
    E = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(E)
    
    #Block: 2
    E = Conv2D(128, (3, 3), activation='relu', padding='same',name='block2_conv1')(E)
    E = Conv2D(128, (3, 3), activation='relu', padding='same',name='block2_conv2')(E)
    E = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(E)

    #Block: 3
    E = Conv2D(256, (3, 3), activation='relu', padding='same',name='block3_conv1')(E)
    E = Conv2D(256, (3, 3), activation='relu', padding='same',name='block3_conv2')(E)
    E = Conv2D(256, (3, 3), activation='relu', padding='same',name='block3_conv3')(E)
    E = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(E)

    #Block: 4
    E = Conv2D(512, (3, 3), activation='relu', padding='same',name='block4_conv1')(E)
    E = Conv2D(512, (3, 3), activation='relu', padding='same',name='block4_conv2')(E)
    E = Conv2D(512, (3, 3), activation='relu', padding='same',name='block4_conv3')(E)
    feat_map = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(E)

    return img_ip, feat_map