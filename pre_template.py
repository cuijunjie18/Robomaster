import cv2
import numpy as np

def show(img,title):
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 返回二极化的模板图片
def Getthresh(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)[1]
    return img


img = cv2.imread('5_cut.png')
thresh = Getthresh(img)

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
temp = img.copy()

"""
for i in range(len(contours)):
    temp = cv2.drawContours(temp,[contours[i]],-1,(0,0,255),5)
    show(temp,'draw')
"""
maxn_contour = 0
maxn_size = 0

for i in range(len(contours)):
    temp_size = cv2.contourArea(contours[i])
    if temp_size > maxn_size:
        maxn_size = temp_size
        maxn_contour = contours[i]
# 切割成矩形
x,y,w,h = cv2.boundingRect(maxn_contour)
temp = temp[y : y + h,x : x + w]
show(temp,'test')
cv2.imwrite('5_cut_solve.png',temp)
