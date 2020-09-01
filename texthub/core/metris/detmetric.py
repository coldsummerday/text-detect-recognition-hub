import numpy as np
from shapely.geometry import Polygon


def plg_get_union(pD:Polygon,pG:Polygon):
    return pD.union(pG).area

def plg_get_intersection(pD:Polygon,pG:Polygon):
    return pD.intersection(pG).area

def plg_get_intersection_over_union(pD:Polygon, pG:Polygon):
    return plg_get_intersection(pD, pG) / plg_get_union(pD, pG)

def polygon_to_bbox(pG:Polygon):
    xmin, ymin, xmax, ymax = pG.boundingBox()
    return xmin,xmax,ymin,ymax

def eval_poly_detect(preds:[[Polygon]],gts:[[Polygon]],thresh = 0.5):
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
                if (inter * 1.0 / union) >= thresh:
                    if gt_id not in cover:
                        flag = True
                        cover.add(gt_id)

            if flag:
                tp += 1.0
            else:
                fp += 1.0
    precision = tp / (tp + fp)
    recall = tp / npos
    hmean = 0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall)
    return dict(
        precision=precision,
        recall=recall,
        hmean=hmean
    )