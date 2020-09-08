import Polygon as plg
import numpy as np
import cv2
from collections import OrderedDict
"""
像素级别的iou计算
"""

def plg_get_union(pD:plg.Polygon,pG:plg.Polygon):
    areaA = pD.area()
    areaB = pG.area()
    return areaA + areaB - plg_get_intersection(pD, pG)

def plg_get_intersection(pD:plg.Polygon,pG:plg.Polygon):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()

def polygon_from_points(points):
    """
    Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    :param points:
    :return:
    """
    p = np.array(points)
    p = p.reshape(p.shape[0] // 2, 2)
    p = plg.Polygon(p)
    return p

def rectangle_to_polygon(rect):
    resBoxes = np.empty([1, 8], dtype='int32')
    resBoxes[0, 0] = int(rect.xmin)
    resBoxes[0, 4] = int(rect.ymax)
    resBoxes[0, 1] = int(rect.xmin)
    resBoxes[0, 5] = int(rect.ymin)
    resBoxes[0, 2] = int(rect.xmax)
    resBoxes[0, 6] = int(rect.ymin)
    resBoxes[0, 3] = int(rect.xmax)
    resBoxes[0, 7] = int(rect.ymax)
    pointMat = resBoxes[0].reshape([2, 4]).T
    return plg.Polygon(pointMat)
def rectangle_to_points(rect):
    points = [int(rect.xmin), int(rect.ymax), int(rect.xmax), int(rect.ymax), int(rect.xmax), int(rect.ymin), int(rect.xmin), int(rect.ymin)]
    return points

def polygon_to_bbox(pG:plg.Polygon):
    xmin,xmax,ymin,ymax = pG.boundingBox()
    return xmin,xmax,ymin,ymax


def eval_poly_detect(preds:[[plg.Polygon]],gts:[[plg.Polygon]],thresh = 0.5):
    tp, fp, npos = 0, 0, 0
    assert len(preds)==len(gts)
    image_nums = len(preds)
    for i in range(image_nums):
        image_preds = preds[i]
        image_gts = gts[i]
        cover = set()
        npos += len(image_gts)
        for pred_id,pred_poly in enumerate(image_preds):
            flag = False
            for gt_id,gt_poly in enumerate(image_gts):
                union = plg_get_union(pred_poly, gt_poly)
                inter = plg_get_intersection(pred_poly, gt_poly)
                if union==0:
                    continue
                if (inter * 1.0 / union) >= thresh:
                    if gt_id not in cover:
                        flag = True
                        cover.add(gt_id)

            if flag:
                tp += 1.0
            else:
                fp += 1.0
    if (tp+fp)==0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    recall = tp / npos
    hmean = 0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall)
    return OrderedDict(
        precision=precision,
        recall=recall,
        hmean=hmean
    )
#     pred_p = plg.Polygon(pred)
#
#     flag = False
#     for gt_id, gt in enumerate(gts):
#         gt = np.array(gt)
#         gt = gt.reshape(gt.shape[0] / 2, 2)
#         gt_p = plg.Polygon(gt)
#
#         union = get_union(pred_p, gt_p)
#         inter = get_intersection(pred_p, gt_p)
#
#         if inter * 1.0 / union >= th:
#             if gt_id not in cover:
#                 flag = True
#                 cover.add(gt_id)
#     if flag:
#         tp += 1.0
#     else:
#         fp += 1.0
#
# tp, fp, npos
# precision = tp / (tp + fp)
# recall = tp / npos
# hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)




