#/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging as log
import os
import cv2
import sys
import  pyclipper
import numpy as np
import torch
this_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(this_path,'../'))

from texthub.apis import init_detector,inference_detector
from texthub.apis import init_recognizer,inference_recognizer

import streamlit as st



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASEDIR = os.path.abspath(os.path.dirname(__file__))

det_config_file = os.path.join(BASEDIR,"../configs/detection/pan/pandetect.py")
det_checkpoint = os.path.join(BASEDIR,"../work_dirs/pan/PAN_epoch_24.pth")


rec_config_file = os.path.join(BASEDIR,"../configs/recognition/fourstagerecogition/tps_resnet_lstm_attention_chi_iter.py")
rec_checkpoint = os.path.join(BASEDIR,"../work_dirs/fourstage_tps_resnet_attention_chinese_iter/iter_300000.pth")
det_model = init_detector(det_config_file, det_checkpoint, device)
rec_model = init_recognizer(rec_config_file, rec_checkpoint, device)

log.basicConfig(
        format='%(asctime)s:%(levelname)s:%(message)s',
         level=log.DEBUG)
def crop_by_poly(img: np.array, points:np.ndarray):
    # 根据多边形裁剪图片，并且返回多边形的外接矩形框
    # size(1,x,2)的ndarray
    #points [4,2]
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    # method 1 smooth region
    cv2.drawContours(mask, [points.astype(int)], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # method 2 not so smooth region
    # cv2.fillPoly(mask, [points], (255))
    res = cv2.bitwise_and(img, img, mask=mask)
    # 希望填充为白色，如果希望填充为黑色则去掉
    ## crate the white background of the same size of original image
    wbg = np.ones_like(img, np.uint8) * 255
    cv2.bitwise_not(wbg, wbg, mask=mask)
    # overlap the resulted cropped image on the white background
    res = wbg + res
    rect = cv2.boundingRect(points)  # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    return cropped

def upload_img():
    src = st.file_uploader(label="上传图片", type=["png", "jpg"])
    src_place = st.empty()
    if src is not None:
        src_place = st.image(src)
    return src, src_place

def text_recognition(src):
    image_data = src.read()

    image_bytes = bytearray(image_data)
    image_array = np.asarray(image_bytes)
    cv2_img_array = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    batch_pred_bbox,score_bbox_list = inference_detector(det_model, cv2_img_array)

    pred_texts = []
    imgs_cropped = []
    for bbox in batch_pred_bbox:
        #应该检测边缘有没有黑色像素再决定要不要添加多边形区域轮廓
        # #是否增大检测面积
        # bbox = addPolygonEdge(bbox)

        croped_img = crop_by_poly(img=cv2_img_array, points=bbox)

        predtext = inference_recognizer(rec_model, croped_img)
        pred_texts.append(predtext)
        imgs_cropped.append(croped_img)
    img = draw_ploygon_img(cv2_img_array,batch_pred_bbox)
    return img, pred_texts, imgs_cropped



def addPolygonEdge(bbox:np.ndarray,distance=1.5):
    """
    bbox:(N,2)
    """
    assert bbox.ndim == 2
    assert bbox.shape[1] == 2
    poly = bbox.copy().astype(np.int)

    # d_i = cv2.contourArea(poly) * (1 - shrink_ratio * shrink_ratio) / cv2.arcLength(poly, True)
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    padded_polygon_pc = pco.Execute(distance)
    # if len(padded_polygon_pc) == 0:
    #     return
    padded_polygon = np.array(padded_polygon_pc)
    return padded_polygon


def draw_ploygon_img(img:np.ndarray,batch_pred_bbox:[np.ndarray]):
    for point in batch_pred_bbox:
        point = point.astype(int)
        cv2.polylines(img, [point], True, (0, 255, 255))
    return img

def display_recognition_result(imgs_cropped, texts):
    contents = []
    for row in texts:
        single_content = 'CONTENT:' + row
        contents.append(single_content)
    st.image(imgs_cropped, caption  = contents)

def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)

    # 创建提示对象
    tip = st.empty()

    # 创建上传的图片对象，获取上传的图片，以及图片占位对象
    src, src_place = upload_img()

    # checkboxes对象
    recognition_flag = st.checkbox('文字识别')
    extraction_flag = st.checkbox('语义提取')

    # 执行按钮
    if st.button("执行"):
        if not src:
            tip.info("请先上传图片")
        elif not recognition_flag:
            tip.info("文字识别为必选")
        else:

            # 文本检测与文本识别
            img, text, imgs_cropped = text_recognition(src)

            # 展示检测结果
            src_place.image(img)

            st.subheader("全字段识别结果")
            # 展示识别结果
            display_recognition_result(imgs_cropped, text)




if __name__ == "__main__":
    main()
