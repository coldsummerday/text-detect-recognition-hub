

### High-level APIs for testing images

#### Synchronous interface
Here is an example of building the model and test given images for text detection.

```python
import os.path as osp
import os
import sys
import cv2
this_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(osp.join(this_path,'../'))
from texthub.apis import init_detector,inference_detector
import torch
config_file = "./configs/testpandetect.py"
checkpoint = "./work_dirs/pan/PAN_epoch_24.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = init_detector(config_file,checkpoint,device)

def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        cv2.line(img_path, tuple(point[0]), tuple(point[1]), color, thickness)
        cv2.line(img_path, tuple(point[1]), tuple(point[2]), color, thickness)
        cv2.line(img_path, tuple(point[2]), tuple(point[3]), color, thickness)
        cv2.line(img_path, tuple(point[3]), tuple(point[0]), color, thickness)
    return img_path

img = "test.jpg"
preds = inference_detector(model,img)

img = cv2.imread(img)

img = draw_bbox(img,preds)
cv2.imshow("s",img)
```


for text recognition:

```python
import os.path as osp
import os
import sys
this_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(osp.join(this_path,'../'))
from texthub.apis import init_recognizer,inference_recognizer
import torch
config_file ="./configs/recognition/aster/aster_tps_resnet11_attention_eng.py"
checkpoint = "./work_dirs/aster_tps_resnet_attion_eng/AsterRecognizer_epoch_20.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = init_recognizer(config_file,checkpoint,device)

img = "testreg.jpg"
print(inference_recognizer(model,img))
```