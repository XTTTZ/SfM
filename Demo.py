import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1=cv.imread('p5.jpg',1)#-1，0，1表示加载方式，0是灰度，1是彩色，-1是alpha通道，默认彩色
img2=cv.imread('p6.jpg',1)
# cv.imshow('p1',img1)#p1是窗口名字，img1是图像变量名
# cv.waitKey(0)#不论多长时间，永远等待键盘输入
# cv.destroyAllWindows()
#方法二显示图片
#plt.imshow(img1[:,:,::-1])#-1是翻转通道
#plt.show()

#对像素进行操作
# a=img1[640,480]#显示像素RGB
# print(a)
# img1[640,480:640]=(0,0,255)#改变像素点
# plt.imshow(img1[:,:,::-1])
# plt.show()
def show_img(img, name):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()
#图像属性
shape=img1.shape
print('shape=',shape)
type=img1.dtype
print('type=',type)
size=img1.size
print('size=',size)


## 第一步：构造sift，求解出特征点和sift特征向量**#*********************************************
sift=cv.xfeatures2d.SIFT_create()#实例化sift
gray1=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)#转为灰度图
gray2=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
#检测关键点并计算，kp是关键点信息，包括位置，尺度，方向信息
#des是关键点描述符，每个关键点对应128个梯度信息的特征向量
kp1,des1=sift.detectAndCompute(gray1,None)
kp2,des2=sift.detectAndCompute(gray2,None)

##############################Matching##################################################
# 第二步：构造BFMatcher()蛮力匹配，匹配sift特征向量距离最近对应组分******************************
print ('SIFT Points Match')

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
# 为了提高检测速度，你可以调用matching函数前，先训练一个matcher。
# 训练过程可以首先使用cv::FlannBasedMatcher来优化，为descriptor建立索引树，这种操作将在匹配大量数据时发挥巨大作用。
# 而Brute-force matcher在这个过程并不进行操作，它只是将train descriptors保存在内存中。



# flann = cv.FlannBasedMatcher(index_params, search_params)#法1快速最近邻搜索算法寻找（用快速的第三方库近似最近邻搜索算法）
bf = cv.BFMatcher()#//法2暴力方法找到点集1中每个descriptor在点集2中距离最近的descriptor；找寻到的距离最小就认为匹配
#                                         //浮点描述子-欧氏距离；二进制描述符-汉明距离。
#                                      //详细描述：在第一幅图像中选取一个关键点然后依次与第二幅图像的每个关键点进行（描述符）距离测试，最后返回距离最近的关键点

matches = bf.knnMatch(des1, des2, k=2)

good = []
points1 = []
points2 = []
################################################################################
#第三步：对匹配的结果按照距离进行筛选**************************************************
for i, (m, n) in enumerate(matches):

    if m.distance < 0.43 * n.distance:#阈值
        good.append(m)
        points1.append(kp1[m.queryIdx].pt)#？？？？？
        points2.append(kp2[m.trainIdx].pt)

#剔除特征点步骤
# （1）根据matches将特征点对齐，将坐标转换为float类型
#
# （2）使用求基础矩阵的方法，findFundamentalMat，得到RansacStatus
#
# （3）根据RansacStatus来删除误匹配点，即RansacStatus[0]=0的点。
points1 = np.float32(points1)
points2 = np.float32(points2)

#RANSAC（RANdom SAmple Consensus)
# 随机抽样一致性，它可以从一组包含“局外点”的观测数据，通过迭代的方式训练最优的参数模型，不符合最优参数模型的被定义为“局外点”。
F, mask = cv.findFundamentalMat(points1, points2, cv.RANSAC)
print (points2)#此时还是未筛选的points，可与后续操作后的points对比
print('F***************\n',F,'\nmask***************\n',mask)

# We select only inlier points
points1 = points1[mask.ravel() == 1]#ravel()方法将数组维度拉成一维数组,筛选上一步的特征点
points2 = points2[mask.ravel() == 1]#注意，特征点较少时，F与mask为空，这里会报错
#print('Points1***************\n',points1,'\nPoints2***************\n',points2)

########################################################################################
#第四步：使用cv2.drawMacthes进行画图操作***************************************************
matches_img=cv.drawMatches(img1,kp1,img2,kp2,good,None,flags=2);#flags表示有几张图片
show_img(matches_img,'Matches')

#**绘制关键点检测结果**#
cv.drawKeypoints(img1,kp1,img1,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.drawKeypoints(img2,kp2,img2,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#图像显示
plt.figure(figsize=(8,6),dpi=100)
plt.imshow(img1[:,:,::-1])
plt.title('SIFT1')
plt.xticks([]),plt.yticks([])
plt.show()

plt.figure(figsize=(8,6),dpi=100)
plt.imshow(img2[:,:,::-1])
plt.title('SIFT2')
plt.xticks([]),plt.yticks([])
plt.show()

