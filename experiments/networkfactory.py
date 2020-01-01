from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Conv2D

class NetworkFactory():
    def __init__(self):
        self.__architectures__ = dict()
        self.__architectures__['mobilenet'] = self.get_mobilenetyolov2

        self.__normalizers__ = dict()
        self.__normalizers__['mobilenet'] = self.normalize_image_to_mobilenet_input

    def supports(self, architecture):
        return architecture in self.__architectures__

    def get_network(self, cfg, optimizer = None, loss = None, weights = None):
        # './weights/mobilenetyolov2try07abitofaugmentation'
        net = cfg.get('net')
        if self.supports(net):
            model = self.__architectures__[net](cfg)
            if optimizer:
                model.compile(optimizer = optimizer, loss = loss)
            if weights:
                model.load_weights(weights)

            return model
        pass

    def get_normalizer(self, cfg):
        net = cfg.get('net')
        if self.supports(net):
            normalizer = self.__normalizers__[net]
            return normalizer

        #TODO: throw exception or return custom normalizer if i decide to store them in the factory

    def get_mobilenetyolov2(self, cfg):
        mobilenetyolov2 = MobileNet(weights='imagenet', include_top=False, input_shape=(cfg.get('image_width'), cfg.get('image_height'), 3))
        mobilenetyolov2.trainable = False
        layers = mobilenetyolov2.layers[:]

        layers.append(
            Conv2D(filters=(cfg.get('boxes') * (4 + 1 + cfg.get('classes'))), kernel_size=(1, 1), padding="same", name="conv_output"))
        layers.append(Reshape(target_shape=(cfg.get('grid_width'), cfg.get('grid_height'), cfg.get('boxes'), 5 + cfg.get('classes')), name="output"))

        mobilenetyolov2 = Sequential(layers=layers, name="yolov2 mobilenetv2")
        # mobilenetyolov2.summary()

        return mobilenetyolov2

    def normalize_image_to_mobilenet_input(self, im):
        im /= 255
        im -= 0.5
        im *= 2.

        return im


