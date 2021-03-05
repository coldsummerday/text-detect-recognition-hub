import lmdb
import os
from tqdm import tqdm
import numpy as np
import cv2
import io


def read_list(filename):
    with open(filename,'r') as file_handler:
        lines = file_handler.readlines()
        lines = [line.rstrip().encode("utf-8").decode("utf-8") for line in lines ]
    return lines

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)
def Ndarray2Bytes(img:np.array):
    img_encode = cv2.imencode('.jpg', img)
    if img_encode[0]==False:
        return None
    str_encode = img_encode[1].tostring()
    img_bytes=io.BytesIO(str_encode).read()
    return  img_bytes
def conver(lmdb_path,img_path,gt_path,line_flag=True):
    """
           if line_flag ==True,390,902,1856,902,1856,1225,390,1225,0,"金氏眼镜"
           Flase:237,48,237,75,322,75,322,48,明天
    """
    image_ids = os.listdir(img_path)
    image_ids = [os.path.splitext(imgid)[0] for imgid in image_ids]
    # check if gt exist
    exist_imgs = []
    gt_path_fmt = os.path.join(gt_path,"gt_{}.txt")
    for imgid in image_ids:
        if os.path.exists(gt_path_fmt.format(imgid)):
            exist_imgs.append(imgid)
    image_ids = exist_imgs


    env = lmdb.open(lmdb_path, map_size=1099511627776)

    cache = {}
    cnt = 1
    for image_id in  tqdm(image_ids):
        data_id = os.path.join(img_path,"{}.jpg".format(image_id))
        data_id = str(data_id.encode('utf-8').decode('utf-8'))
        #打开gt,依次裁剪图片
        with open(os.path.join(gt_path,"gt_{}.txt".format(image_id))) as file_handler:
            lines  = file_handler.readlines()
        ori_img = cv2.imread(data_id)
        for index,gt_line in  enumerate(lines):
            gt_line= gt_line.rstrip()
            line_params = gt_line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
            text_flag,text_label = _get_lable(line_params,line_flag=line_flag)
            if not text_flag or  text_label=="###" or  text_label=="nan":
                continue
            try:
                cropimg = CropImg(ori_img,line_params[:8])
                img_bytes = Ndarray2Bytes(cropimg)
                if img_bytes==None:
                    continue
            except ValueError:
                continue
            except cv2.error:
                continue
            #img 转文件流

            imageKey = 'image-%09d'.encode() % cnt
            labelKey = 'label-%09d'.encode() % cnt
            cache[imageKey] = img_bytes
            cache[labelKey] = text_label.encode()

            cnt+=1
            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}

    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


            # #这里应该是每一张小图一个data_id
            # per_img_data_id = os.path.join(img_path,"{}_{}.jpg".format(image_id,index))
            # with env.begin(write=True) as lmdb_writer:
            #     lmdb_writer.put(per_img_data_id.encode(),
            #                     pickle.dumps(value), db=db_extra)
            # with env.begin(write=True) as image_writer:
            #     image_writer.put(per_img_data_id.encode(), img_bytes, db=db_image)

    env.close()

def Ndarray2Bytes(img:np.array):
    img_encode = cv2.imencode('.jpg', img)
    if img_encode[0]==False:
        return None
    str_encode = img_encode[1].tobytes()
    img_bytes=io.BytesIO(str_encode).read()
    return  img_bytes

def _get_lable(label_line_params:list,line_flag=True):
    if line_flag:
        text_tag = label_line_params[8:]
        text_label = label_line_params[9:]
        #去除引号
        text_label = text_label[1:-1]
        if text_tag=='1':
            text_tag=False
        else:
            text_tag = True
    else:
        text_label = label_line_params[8:]
        if text_label=="*" or text_label=='###' or text_label=="nan":
            text_tag = False
        else:
            text_tag = True
    return text_tag,"".join(text_label)


def CropImg(oriimg:np.array,polygon_list:[str],edge =1):
    positions = polygon_list
    positions = [int(i) for i in positions]
    x_list = positions[0::2]
    y_list = positions[1::2]
    min_x = min(x_list)
    max_x = max(x_list)
    min_y =min(y_list)
    max_y = max(y_list)
    #roi = im[y1:y2, x1:x2]
    crop_img = oriimg[min_y:max_y,min_x:max_x]
    return  crop_img

def CropByPoly(img: np.array, points:list):
    # 根据多边形裁剪图片，并且返回多边形的外接矩形框
    # size(1,x,2)的ndarray
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    # method 1 smooth region
    cv2.drawContours(mask, points, -1, (255, 255, 255), -1, cv2.LINE_AA)
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

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)


def parse_args():
    import  argparse
    parser = argparse.ArgumentParser(description='create icdar dataset to lmdb recognition dataset')
    parser.add_argument('--data_dir', help='the data_dir includes two dir:imgs,gts')
    parser.add_argument('--save_dir',help="where the lmdb to save ")
    parser.add_argument(
        '--line_flag',
        action='store_true',
        help='whether the dataset is icdar17ctw gt line include hard flag in line[8].')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    mkdir(args.save_dir)

    conver(args.save_dir,
           os.path.join(args.data_dir, "imgs"),
           os.path.join(args.data_dir, "gts"),
           line_flag=args.line_flag)

if __name__ == "__main__":
    main()
