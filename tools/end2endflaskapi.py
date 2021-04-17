#/usr/bin/env python
# -*- coding: UTF-8 -*-
import base64
import io
import json
import logging as log
import os
import cv2
import sys

import numpy as np
import torch
import werkzeug
from flask import Flask
from flask_restful import Api, Resource, reqparse

this_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(this_path,'../'))

from texthub.apis import init_detector,inference_detector
from texthub.apis import init_recognizer,inference_recognizer



#用tornado 部署 flask
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.wsgi import WSGIContainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASEDIR = os.path.abspath(os.path.dirname(__file__))

det_config_file = os.path.join(BASEDIR,"../configs/detecttion/pan/pandetect.py")
det_checkpoint = os.path.join(BASEDIR,"../work_dirs/pan/PAN_epoch_24.pth")


rec_config_file = os.path.join(BASEDIR,
                               "../configs/receipt/recognition/experiments/fourstagerecogition/tps_vgg_lstm_attention.py")
rec_checkpoint = os.path.join(BASEDIR,"../work_dirs/tps_vgg_lstm_attention/FourStageModel_epoch_300.pth")


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

class OCRResource(Resource):
    image_parser = reqparse.RequestParser()
    image_parser.add_argument('image', type=werkzeug.datastructures.FileStorage, location='files', required=True,
                              help="Can't find image parameter")
    det_model = init_detector(det_config_file, det_checkpoint, device)
    rec_model = init_recognizer(rec_config_file, rec_checkpoint, device)

    def post(self):
        args = self.image_parser.parse_args()
        image_file = args.image
        #save image file
        image_name = image_file.filename
        image_data = image_file.read()

        image_bytes = bytearray(image_data)
        image_array = np.asarray(image_bytes)
        cv2_img_array = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        det_result,score_bbox_list = inference_detector(self.det_model,cv2_img_array)
        result_json_list = []
        for bbox in det_result:
            croped_img = crop_by_poly(img=cv2_img_array, points=bbox)
            x1,y1 = bbox[0]
            x2,y2 = bbox[1]
            x3,y3 = bbox[2]
            x4,y4 = bbox[3]

            predtext = inference_recognizer(self.rec_model,croped_img)
            result = dict(
                x1=int(x1),y1=int(y1),x2=int(x2),y2=int(y2),x3=int(x3),y3=int(y3),x4=int(x4),y4=int(y4),predtext=predtext
            )
            result_json_list.append(result)
        return result_json_list



if __name__ == '__main__':
    app = Flask(__name__)
    app.config.update(RESTFUL_JSON=dict(ensure_ascii=False))
    api = Api(app)
    api.add_resource(OCRResource,'/ocr')
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(5000,"0.0.0.0")
    IOLoop.instance().start()



