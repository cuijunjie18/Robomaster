import cv2
import matplotlib.pyplot as plt
# 读取图像
image = cv2.imread('test.png')
# 计算图像的HSV直方图
hist = cv2.calcHist([image], [0, 1, 2], None, [180, 255, 255], [0, 180, 0, 255, 0, 255])
# 归一化直方图
cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
# 显示直方图
plt.imshow(hist, interpolation='nearest')
plt.title('HSV Histogram')
plt.show()