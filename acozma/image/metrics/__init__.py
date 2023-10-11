def iou(bbox1, bbox2):
    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])

    intersection_area = max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    return intersection_area / float(bbox1_area + bbox2_area - intersection_area)


def precision(tp, fp):
    return 0 if tp + fp == 0 else tp / (tp + fp)


def recall(tp, fn):
    return 0 if tp + fn == 0 else tp / (tp + fn)


def f1(precision, recall):
    if precision + recall == 0:
        return 0
    else:
        return 2 * precision * recall / (precision + recall)
