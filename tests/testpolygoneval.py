import os.path as osp
import os
import sys
import numpy as np
from PIL import Image,ImageDraw
import torchvision.transforms as transforms
loader = transforms.Compose([
    transforms.ToTensor()])
unloader = transforms.ToPILImage()
this_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(osp.join(this_path,'../'))
from texthub.modules import build_detector
from texthub.utils import Config
import torch
from texthub.core.evaluation.polygoneval import eval_poly_detect
from texthub.datasets import build_dataset
from texthub.core.utils.checkpoint import load_checkpoint
import Polygon as plg
path = "./work_dirs/pan/PAN_epoch_24.pth"
config_file = "./configs/testpandetect.py"
cfg = Config.fromfile(config_file)
train_dataset = build_dataset(cfg.data.val)
model = build_detector(cfg.model)
load_checkpoint(model, path,map_location=torch.device('cpu'))
model.eval()
batch_size = 16
train_data_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size,
                num_workers=0,
                shuffle=True,
                pin_memory=True)

index = 0
def polygon2points(poly:plg.Polygon):
    points = np.array(poly.contour(0),dtype="uint8")
    points= points.reshape((-1, 1, 2))
    return points

def tensor2cv2(tensor):
    #permute(1, 2, 0).numpy()
    return (tensor.detach().numpy().transpose(1, 2, 0)*256).astype('uint8')

def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().transpose((1, 2, 0))
    return img

def tensor2poly(gt_polys:torch.Tensor):
    #(b,150,4,2)
    results = []
    for array in gt_polys:
        image_polys = []
        for points in array:
            if points[0,0]!=0:
                poly_gon = plg.Polygon(points.cpu().numpy())
                image_polys.append(poly_gon)
        results.append(image_polys)
    return results

preds = []
gts = []
for index ,data in enumerate(train_data_loader):
    imgs = data['img'].clone()
    b = model(data, return_loss=False)
    polys = model.postprocess(b)
    preds.extend(polys)
    gts.extend(tensor2poly(data['gt_polys']))
    # gts = tensor2poly(data['gt_polys'])
    # print(eval_poly_detect(polys, gts))
    if index>5:
        break
    # for i in range(batch_size):
    #     img =  unloader(imgs[i].cpu().clone())
    #     pred_polys = gts[i]
    #     for poly in pred_polys:
    #         # xmin, xmax, ymin, ymax = poly.boundingBox()
    #         # xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
    #         # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    #         # points = polygon2points(poly)
    #         # cv2.fillPoly(img, points, 255)
    #         # points = points.reshape((-1,1,2))
    #         # my_img = np.zeros((640, 640, 3), dtype = "uint8")
    #         # try:
    #         # cv2.fillPoly(img,points,True,(0,255,255))
    #         # except:
    #         #     continue
    #         draw = ImageDraw.Draw(img)
    #         draw.polygon(poly[0], outline=(255, 0, 0))
    #         # draw.rectangle([(xmin,ymin),(xmax,ymax)],outline=(0,255,0))
    #     img.save("./testimgs/gt_{}.jpeg".format(i*index))
    #     #cv2.imwrite("./testimgs/{}.jpeg".format(i*index),img)

print(eval_poly_detect(preds, gts))

# data = train_data_loader.__iter__().__next__()

# draw = ImageDraw.Draw(img)
# draw.polygon(preds[0],outline=(255,0,0))
#
#
#







