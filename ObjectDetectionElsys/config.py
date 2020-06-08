import numpy as np

from ObjectDetectionElsys.networkfactory import NetworkFactory

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

    def parse_line(self, line):
        data = line.split('=')
        if len(data) != 2:
            print('Warning: Skipped a line not in format [attribute]=[value]')
            print(line)
            return None, None

        attr = data[0].strip()
        value = data[1].strip()

        return attr, value

    def parse_cfg(self, cfg_path):
        with open(cfg_path) as f:
            lines = f.readlines()
            index = 0
            count = len(lines)
            while index < count:
                line = lines[index].strip()

                if line == "[net]":
                    self.parse_net(lines[index + 1:])
                    break

                if line == '[optimizer]':
                    index += 1
                    line = lines[index]
                    attr, data = self.parse_line(line)

                    if attr == 'name':
                        self.set('optimizer', data)
                    else:
                        print('Warning: First line after [optimizzer] must be name')
                    index += 1
                    continue

                if line == "" or line[0] == '#' or line[0] == '[':
                    index += 1
                    continue

                attr, value = self.parse_line(line)
                if attr == None:
                    index += 1
                    continue

                if attr == 'anchors':
                    self.set(attr, self.parse_anchors(value))
                else:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)

                    self.set(attr, value)

                index += 1

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



if __name__ == '__main__':
    main()