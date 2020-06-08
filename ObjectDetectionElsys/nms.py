from ObjectDetectionElsys import calculate_IoU

def nms(cfg, objects):
    if len(objects) == 0:
        return []

    objects = sorted(objects, key=lambda obj: obj.conf, reverse=True)

    result = [objects[0]]
    del objects[0]

    for i, obj in enumerate(objects):
        for res in result:
            if calculate_IoU(obj, res) > cfg.get('nms_threshhold'):
                del objects[i]
                break
        else:
            result.append(obj)
            del objects[i]

    return result

def group_nms(cfg, objects):
    keys = set(map(lambda obj: obj.name, objects))
    groups = dict()

    for key in keys:
        groups[key] = []
    for obj in objects:
        groups[obj.name].append(obj)

    result = []
    for key in keys:
        result.extend(nms(cfg, groups[key]))

    return result

