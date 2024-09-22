import cv2
import numpy as np

# RGB (76,212,234)
# 可乐瓶盖 BGR = (71,31,200)
# r:184, g:84, b:183

# 构建展示函数
def show(img,title):
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#根据给定HSV图像确定上下界HSV
def cvt_hsv(hsv):
    b = hsv[0][0][0]
    l_hsv = np.uint8([[[b - 50,100,100]]])
    h_hsv = np.uint8([[[b + 50,255,255]]])
    ans = [l_hsv,h_hsv]
    return ans

# 终端输入图片路径
path = "test.png"

# 读取图片
img = cv2.imread(path)

track_color = np.uint8([[[250,237,46]]])
track_hsv = cv2.cvtColor(track_color,cv2.COLOR_BGR2HSV)
print(track_hsv)

#以hsv设定hsv颜色空间对应物体的阈值
l_hsv,h_hsv = cvt_hsv(track_hsv)
print(l_hsv,h_hsv)

hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

#根据阈值建立掩膜(可以理解为返回接收了一个阈值图像,即满足特定阈值范围的才显示)
mask = cv2.inRange(hsv,l_hsv,h_hsv)

#难点2:对原图像与掩膜进行位运算
res = cv2.bitwise_and(img,img,mask = mask)

#out = np.hstack([frame,mask,res])
cv2.imshow('Origin',img)
cv2.imshow('Mask',mask)
cv2.imshow('Track',res)

cv2.waitKey(0)
cv2.destroyAllWindows()
