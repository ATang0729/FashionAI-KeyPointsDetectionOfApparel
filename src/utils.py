import torch
import cv2
import numpy as np


def bbox_iou(anchor_bboxes, gt_bboxes):
    """计算anchor_bboxes(锚框）与gt_bboxes（ground truth boxes）的IoU

    :param anchor_bboxes: [x1, y1, x2, y2]
    :param gt_bboxes: [x1, y1, x2, y2]
    :return:
    """
    N = anchor_bboxes.size(0)
    M = gt_bboxes.size(0)
    lb = torch.max(anchor_bboxes[:, None, :2], gt_bboxes[:, :2])  # [N, M, 2]
    rb = torch.min(anchor_bboxes[:, None, 2:], gt_bboxes[:, 2:])  # [N, M, 2]
    wh = (rb - lb + 1).clamp(min=0)  # [N, M, 2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    area1 = ((anchor_bboxes[:, 2] - anchor_bboxes[:, 0]) + 1) * (
            (anchor_bboxes[:, 3] - anchor_bboxes[:, 1]) + 1)  # [N,]
    area2 = ((gt_bboxes[:, 2] - gt_bboxes[:, 0]) + 1) * ((gt_bboxes[:, 3] - gt_bboxes[:, 1]) + 1)  # [M,]
    iou = intersection / (area1[:, None] + area2 - intersection)
    return iou


def bbox_nms(bboxes, scores, threshold=0.5, mode='union', topk=5):
    """Non maximum suppression.

    非极大值抑制，简单说就是将得分低、与得分高的框有较大重叠的框进行删除，保留最优的框

    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) bbox scores, sized [N,].
      threshold: (float) overlap threshold.
      mode: (str) 'union' or 'min'.
    Returns:
      keep: (tensor) selected indices.
    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    """
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    keep = keep[:topk]
    return torch.LongTensor(keep)


def draw_bbox(image, bboxes, probs, save_path, gt_bboxes=None):
    '''将框画在图像上，同时在框周围画上置信度（probs），也可以画出gt_bboxes

    :param image: 
    :param bboxes: [[x1, y1, x2, y2], ...]
    :param labels: string list
    :param probs: float array
    :param save_path: string end with file name
    :return: None
    '''
    alpha = 0.5
    color = (0, 255, 0)
    thick = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    overlay = image.copy()
    for bbox, prob in zip(bboxes, probs):
        label_txt = 'Prob: %.2f' % prob
        x1, y1, x2, y2 = np.round(bbox).astype(np.int)
        overlay = cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thick)
        txt_size = cv2.getTextSize(label_txt, font, font_scale, thick)
        overlay = cv2.rectangle(overlay, (x1, y1 - txt_size[0][1]), (x1 + txt_size[0][0], y1), color, cv2.FILLED)
        overlay = cv2.putText(overlay, label_txt, (x1, y1), font, font_scale, (255, 255, 255), thick, cv2.LINE_AA)
    if gt_bboxes is not None:
        for bbox in gt_bboxes:
            x1, y1, x2, y2 = np.round(bbox).astype(np.int)
            overlay = cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), thick)
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    cv2.imwrite(save_path, image)


def draw_keypoint_with_caption(image, keypoint, text):
    """Draw a keypoint on the image with a caption added below the image.

    将关键点画在图像上，并在图像下方加上文字描述

    :param image:
    :param keypoint: [x, y]
    :param text: string
    :return: image
    """
    # Set the transparency of the overlaid image
    alpha = 0.5
    # Set the color of the keypoint to green
    color1 = (0, 255, 0)
    # Set the thickness of the lines used for drawing to 2 pixels
    thick = 2
    # Set the length of lines used to create the crosshair around the keypoint's position
    l = 5
    # Set the font used for adding the caption to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Set the scale of the font used for adding the caption to the image
    font_scale = 0.5
    overlay = image.copy()
    x, y = keypoint
    # Draw two lines to create a crosshair around the keypoint's position
    overlay = cv2.line(overlay, (x - l, y - l), (x + l, y + l), color1, thick)
    overlay = cv2.line(overlay, (x - l, y + l), (x + l, y - l), color1, thick)
    # Add the caption to the image
    overlay = cv2.putText(overlay, text, (0, image.shape[0]), font, font_scale, (0, 0, 0), thick, cv2.LINE_AA)
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image


color_palette = [(136, 112, 246),
                 (49, 136, 219),
                 (49, 156, 173),
                 (49, 170, 119),
                 (122, 176, 51),
                 (164, 172, 53),
                 (197, 168, 56),
                 (244, 154, 110),
                 (244, 121, 204),
                 (204, 101, 245)]  # husl


def draw_keypoints(image, keypoints, gt_keypoints=None):
    """将关键点画在图像上，gt_keypoints用于画Ground Truth的点。

    :param image:
    :param keypoints: [[x, y, v], ...]
    :return:
    """
    # 初始化变量：初始化了alpha、颜色、线条粗细等一些常量，一个字体和字体大小。
    alpha = 0.8
    color1 = (0, 255, 0)
    color2 = (0, 0, 255)
    thick = 2
    l = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    overlay = image.copy()
    # 根据是否有提供Ground Truth的点，选择绘制方式
    # 如果没有，遍历每个关键点，将结果按顺序绘制到图像上，如果该点"v"(visible)的值大于0，则按照给定的颜色和大小画出十字线，表示该点的位置
    if gt_keypoints is None:
        for i, kpt in enumerate(keypoints):
            x, y, v = kpt
            if v > 0:
                overlay = cv2.line(overlay, (x - l, y - l), (x + l, y + l), color_palette[i % len(color_palette)],
                                   thick)
                overlay = cv2.line(overlay, (x - l, y + l), (x + l, y - l), color_palette[i % len(color_palette)],
                                   thick)
    # 如果有Ground Truth的点，那么将该点的信息和当前关键点的信息一块儿绘制出来，会在关键点所在位置绘制一个编号和十字线，
    # 并以不同颜色绘制Ground Truth的点和编号，一个红色十字线和一个蓝色编号。
    if gt_keypoints is not None:
        for k in range(len(keypoints)):
            gtx, gty, gtv = gt_keypoints[k]
            x, y, v = keypoints[k]
            if gtv > 0:
                overlay = cv2.line(overlay, (x - l, y - l), (x + l, y + l), color1, thick)
                overlay = cv2.line(overlay, (x - l, y + l), (x + l, y - l), color1, thick)
                overlay = cv2.putText(overlay, str(k), (x, y), font, font_scale, color1, thick, cv2.LINE_AA)
                overlay = cv2.line(overlay, (gtx - l, gty - l), (gtx + l, gty + l), color2, thick)
                overlay = cv2.line(overlay, (gtx - l, gty + l), (gtx + l, gty - l), color2, thick)
                overlay = cv2.putText(overlay, str(k), (gtx, gty), font, font_scale, color2, thick, cv2.LINE_AA)
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image


def draw_heatmap(image, heatmap):
    """将Heatmap画在指定的图像上

    :param image:
    :param heatmap:
    :param save_path:
    :return:
    """
    hp_max = np.amax(heatmap)
    scale = 1
    if hp_max != 0:
        scale = 255 // hp_max
    heatmap = (heatmap * scale).astype(np.uint8)
    alpha = 0.7
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    fin = cv2.addWeighted(heatmap_img, alpha, image, 1 - alpha, 0)
    return fin


def normalized_error(preds, targets, widths):
    """根据预测的关键点和真实的关键点计算评价指标

    :param preds: [[x, y, v], ...]
    :param targets: [[x, y, v], ...]
    :param widths: [[w1], [w2], ...]
    :return:
    """
    dist = preds[:, :2] - targets[:, :2]
    dist = np.sqrt(dist[:, 0] ** 2 + dist[:, 1] ** 2)
    targets = np.copy(targets)
    targets[targets < 0] = 0
    if np.sum(targets[:, 2]) == 0:
        return 0
    ne = np.sum(dist / widths * targets[:, 2]) / np.sum(targets[:, 2])
    return ne


if __name__ == '__main__':
    from kpda_parser import KPDA
    from src.config import Config
    import cv2

    config = Config('trousers')
    db_path = '/home/storage/lsy/fashion/FashionAI_Keypoint_Detection/'
    kpda = KPDA(config, db_path, 'train')
    for idx in range(kpda.size()):
        img = cv2.imread(kpda.get_image_path(idx))  # BGR
        # bboxes = [kpda.get_bbox(idx)]
        # probs = [1.]
        # draw_bbox(img, bboxes, probs, '/home/storage/lsy/fashion/tmp/%d.png' % idx)
        keypoints = kpda.get_keypoints(idx)
        if np.all(keypoints[:, 2] > 0):
            print(keypoints)
            draw = draw_keypoints(img, keypoints)
            cv2.imwrite('/home/storage/lsy/fashion/tmp/%d.png' % idx, draw)
            break
