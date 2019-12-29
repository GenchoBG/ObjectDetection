from tensorflow.keras.applications import MobileNet
import numpy as np

class NetworkFactory():
    def __init__(self):
        self.__architectures__ = dict()
        self.__architectures__['mobilenet'] = MobileNet

    def supports(self, architecture):
        return architecture in self.__architectures__


networkfactory = NetworkFactory()


class Config():
    def __init__(self, cfg_path=None):
        self.__data__ = dict()

        if cfg_path:
            self.parse_cfg(cfg_path)

    def set(self, attr, value):
        self.__data__[attr] = value

    def get(self, attr):
        return self.__data__[attr]

    def parse_cfg(self, cfg_path):
        with open(cfg_path) as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                line = line.strip()

                if line == "[net]":
                    self.parse_net(lines[index + 1:])
                    break

                if line == "" or line[0] == '#' or line[0] == '[':
                    continue

                data = line.split('=')
                if len(data) != 2:
                    print('Warning: Skipped a line not in format [attribute]=[value]')
                    print(line)
                    continue

                attr = data[0].strip()
                value = data[1].strip()

                if attr == 'anchors':
                    self.set(attr, self.parse_anchors(value))
                else:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)

                    self.set(attr, value)

            self.post_calculations()

    def post_calculations(self):
        image_width = self.get('image_width')
        image_height = self.get('image_height')

        grid_width = int(image_width / 32)  # 13
        grid_height = int(image_height / 32)  # 13

        cell_width = image_width / grid_width
        cell_height = image_height / grid_height

        self.set('grid_width', grid_width)
        self.set('grid_height', grid_height)
        self.set('cell_width', cell_width)
        self.set('cell_height', cell_height)

    def parse_net(self, lines):
        if len(lines) == 1:
            # parse predefined architecture
            line = lines[0]

            data = line.split('=')
            if len(data) != 2:
                print('Warning: Line not in format predefined=[architecture]')
                print(line)
                return  # TODO: Exception

            architecture = data[1].strip()
            if not networkfactory.supports(architecture):
                print('Warning: Architecture not supported')

            self.set('net', architecture)
        else:
            # parse custom architecture
            pass

    def parse_anchors(self, anchors):
        anchors = np.array([float(a.strip()) for a in anchors.split(',')], dtype=np.float32)
        boxes = self.get('boxes')

        rows = int(boxes)
        cols = int(anchors.shape[0] / rows)

        anchors = anchors.reshape((rows, cols))

        return anchors


def main():
    cfg_path = r"C:\Users\Gencho\Desktop\ObjectDetection\experiments\mobilenetyolov2-voc.cfg"
    cfg = Config(cfg_path)

main()