import cv2
import numpy as np
img = cv2.imread('univ1.jpg',1)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
surf = cv2.xfeatures2d_SURF(800) #SURF Hessian 的阈值
kp, des = surf.detectAndCompute(img) #寻找关键点
print( len(kp)) #打印关键点个数
cv2.drawKeypoints(img,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #绘制关键点
cv2.imshow("img",img)
cv2.waitKey()
cv2.destroyAllWindows()
