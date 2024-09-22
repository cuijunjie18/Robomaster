import cv2
import numpy as np


# 构建展示函数
def show(img,title):
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#根据给定HSV图像确定上下界HSV
def cvt_hsv(hsv):
    b = hsv[0][0][0]
    l_hsv = np.uint8([[[b-10,100,100]]])
    h_hsv = np.uint8([[[b+10,255,255]]])
    ans = [l_hsv,h_hsv]
    return ans

# 返回二极化的模板图片
def Getthresh(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)[1]
    return img

# 响应鼠标事件
def mouse_callback(event, x, y, flags, userdata):
    # 如果鼠标左键点击，则输出横坐标和纵坐标
    if event == cv2.EVENT_LBUTTONDOWN:
        point_list.append([x,y])

# 导入特征模板
template_contours = []
for i in range(5):
    template_path = 'templates/' + str(i + 1) + '_cut_solve.png'
    template = cv2.imread(template_path)
    template = Getthresh(template)
    template_contours.append(cv2.findContours(template,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0])

"""
template_2 = cv2.imread('1_cut.png')
template_2 = Getthresh(template_2)
cnt2,hierarchy = cv2.findContours(template_2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
"""

# show(template_2,'Test')

# 终端输入图片路径
path = "images/1018.jpg"
# print("Input the path of the image:")
# path = input()

# 读取并简单处理图片
img = cv2.imread(path)
img = cv2.GaussianBlur(img,(5,5),0)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
print(thresh.shape)

# 建立点击列表
point_list = []
cv2.namedWindow('Point Coordinates')
cv2.setMouseCallback('Point Coordinates', mouse_callback)

while True:
    cv2.imshow('Point Coordinates', thresh)
    k = cv2.waitKey(1) & 0xFF
    # 按esc键退出
    if k == 27:
        break
    if len(point_list) == 1:
        break

cv2.destroyAllWindows()
x = point_list[0][0]
y = point_list[0][1]

print(x,y)
cut_thresh = thresh.copy()[y - 60: y + 60,x - 60 : x + 60]
print(cut_thresh.shape)
show(cut_thresh,'cut_thresh')

# 获取模板外轮廓，且仅要面积最大的，即数字的轮廓
contours,hierarchy = cv2.findContours(cut_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
temp = img.copy()[y - 60: y + 60,x - 60 : x + 60]
maxn_contours = 0
maxn_size = 0
for i in range(len(contours)):
    temp_size = cv2.contourArea(contours[i])
    if temp_size > maxn_size:
        maxn_size = temp_size
        maxn_contours = contours[i]

x,y,w,h = cv2.boundingRect(maxn_contours)
temp = cut_thresh[y : y + h,x : x + w]
show(temp,'Test')

# 匹配最佳数字
minn = 9999
ans = -1
for i  in range(5):
    res = cv2.matchShapes(template_contours[i][0],maxn_contours,1,0.0)
    if res < minn and res < 0.55:
        minn = res
        ans = i + 1

if ans == -1:
    print("NULL")
else:
    print(f"the num of the robot is {ans}")

