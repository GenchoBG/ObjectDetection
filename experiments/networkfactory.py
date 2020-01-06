from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Conv2D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

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

            model.compile(optimizer = optimizer if optimizer else self.get_optimizer(cfg), loss = loss)

            if weights:
                model.load_weights(weights)

            return model
        pass

    def get_optimizer(self, cfg):
        optimizer = cfg.get('optimizer')
        if not optimizer:
            print('Warning: optimizer not specified in .cfg')

        optimizer = optimizer.lower()
        if optimizer == 'adam':
            lr = cfg.get('learning_rate')
            beta_1 = cfg.get('beta_1')
            beta_2 = cfg.get('beta_2')
            epsilon = cfg.get('epsilon')
            decay = cfg.get('decay')


            return Adam(lr = lr, beta_1 = beta_1, beta_2 = beta_2, epsilon = epsilon, decay = decay)

        if optimizer == 'sgd':
            lr = cfg.get('learning_rate')
            momentum = cfg.get('momentum')
            decay = cfg.get('decay')

            return SGD(lr = lr, momentum = momentum, decay = decay)

        if optimizer == 'rmsprop':
            lr = cfg.get('learning_rate')
            rho = cfg.get('rho')
            epsilon = cfg.get('epsilon')
            decay = cfg.get('decay')

            return RMSprop(lr = lr, rho = rho, epsilon = epsilon, decay = decay)


        print(f'Warning: unrecognised optimizer "{optimizer}"')

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


