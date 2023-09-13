import keras

from keras.models import *
from keras.layers import *

from VGG16 import get_vgg16_encoder
from model_utils import c_acc, c_loss

class SegNet():
    def __init__(self, lr, img_shape):

        self.lr = lr
        self.img_shape = img_shape

    def segnet_decoder(self, feat_map, n_up):

        assert n_up >= 2

        #decoder: 1
        D = feat_map
        D = UpSampling2D((2, 2))(D)
        D = ZeroPadding2D((1, 1))(D)
        D = Conv2D(512, (3, 3), padding='valid')(D)
        D = BatchNormalization()(D)

        #decoder: 2
        D = UpSampling2D((2, 2))(D)
        D = ZeroPadding2D((1, 1))(D)
        D = Conv2D(256, (3, 3), padding='valid')(D)
        D = BatchNormalization()(D)

        #decoder: 3
        for _ in range(n_up-2):
            D = UpSampling2D((2, 2))(D)
            D = ZeroPadding2D((1, 1))(D)
            D = Conv2D(128, (3, 3), padding='valid')(D)
            D = BatchNormalization()(D)

        #decoder: 4
        D = UpSampling2D((2, 2))(D)
        D = ZeroPadding2D((1, 1))(D)
        D = Conv2D(64, (3, 3), padding='valid')(D)
        D = BatchNormalization()(D)
        
        D = Conv2D(1, 1, padding='same', activation='sigmoid')(D)
        img_op = Reshape(((D.shape[1] * D.shape[2]), D.shape[3]))(D)

        return img_op

    def _segnet(self, encoder, ip_height, ip_width, channels):
        
        img_ip, feat_map = encoder(ip_height, ip_width, channels)
        img_op = self.segnet_decoder(feat_map, n_up = 3)
        model = Model(inputs = img_ip, outputs = img_op)

        return model

    def vgg16_segnet(self, ip_height, ip_width, channels=3):
        
        model = self._segnet(get_vgg16_encoder, ip_height, ip_width, channels)
        model.model_name = "VGG16_SegNet"

        return model

    def resnet50_segnet(self):
        pass

    def mobilenet_segnet(self):
        pass 

    def init_model(self, backbone):
        
        assert len(self.img_shape) == 3
        ip_height, ip_width, channels = self.img_shape
        
        if backbone == 0:
            model = self.vgg16_segnet(ip_height, ip_width, channels)
        elif backbone == 1:
            pass
        elif backbone == 2:
            pass
        else:
            pass

        opt = keras.optimizers.RMSprop(lr = self.lr, rho=0.9, epsilon=1e-08)

        #custom loss and accuracy are needed to filter out the void labels in the CDnet2014_dataset
        model.compile(loss= c_loss, optimizer=opt, metrics=[c_acc], sample_weight_mode = 'temporal')

        return model