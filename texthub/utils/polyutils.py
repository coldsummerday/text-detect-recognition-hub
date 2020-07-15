import  numpy as np
import  cv2
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