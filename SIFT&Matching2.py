import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import structure
from mpl_toolkits.mplot3d import Axes3D
from Camfixing import camMat

img1=cv.imread('E:\\2_Univercity\\3_3_Abroad\\2_RoboticUIUC\\3_Paper\\git\\3Dreconstruction\\imgs\\dinos\\ (1).ppm',1)#-1，0，1表示加载方式，0是灰度，1是彩色，-1是alpha通道，默认彩色
img2=cv.imread('E:\\2_Univercity\\3_3_Abroad\\2_RoboticUIUC\\3_Paper\\git\\3Dreconstruction\\imgs\\dinos\\ (4).ppm',1)
def show_img(img, name):
    cv.namedWindow(name, 0)
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

## 第一步：构造sift，求解出特征点和sift特征向量**#*********************************************
sift=cv.SIFT_create()#实例化sift
gray1=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)#转为灰度图
gray2=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
#检测关键点并计算，kp是关键点信息，包括位置，尺度，方向信息
#des是关键点描述符，每个关键点对应128个梯度信息的特征向量
kp1,des1=sift.detectAndCompute(gray1,None)
kp2,des2=sift.detectAndCompute(gray2,None)

##############################Matching##################################################
# 第二步：构造BFMatcher()蛮力匹配，匹配sift特征向量距离最近对应组分******************************
#print ('SIFT Points Match')

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
'''
为了提高检测速度，你可以调用matching函数前，先训练一个matcher。
训练过程可以首先使用cv::FlannBasedMatcher来优化，为descriptor建立索引树，这种操作将在匹配大量数据时发挥巨大作用。
而Brute-force matcher在这个过程并不进行操作，它只是将train descriptors保存在内存中。 
'''




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

    if m.distance < 0.9 * n.distance:#阈值,不同照片这里可能并不是最佳阈值
        good.append(m)
        points1.append(kp1[m.queryIdx].pt)#？？？？？
        points2.append(kp2[m.trainIdx].pt)
'''
    剔除特征点步骤
（1）根据matches将特征点对齐，将坐标转换为float类型

（2）使用求基础矩阵的方法，findFundamentalMat，得到RansacStatus

（3）根据RansacStatus来删除误匹配点，即RansacStatus[0]=0的点。
'''

points1 = np.float32(points1)
points2 = np.float32(points2)

#RANSAC（RANdom SAmple Consensus)
# 随机抽样一致性，它可以从一组包含“局外点”的观测数据，通过迭代的方式训练最优的参数模型，不符合最优参数模型的被定义为“局外点”。
F, mask = cv.findFundamentalMat(points1, points2, cv.RANSAC)
# print (points2)#此时还是未筛选的points，可与后续操作后的points对比
# print('F***************\n',F,'\nmask***************\n',mask)

# We select only inlier points
points1 = points1[mask.ravel() == 1]#ravel()方法将数组维度拉成一维数组,筛选上一步的特征点
points2 = points2[mask.ravel() == 1]#注意，特征点较少时，F与mask为空，这里会报错
#print('Points1***************\n',points1,'\nPoints2***************\n',points2)

########################################################################################
#第四步：使用cv2.drawMacthes进行画图操作***************************************************
matches_img=cv.drawMatches(img1,kp1,img2,kp2,good,None,flags=2);#flags表示有几张图片
show_img(matches_img,'Matches')
'''
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
'''
################################################################################
# camera matrix from calibration
#已知内参数矩阵，基础矩阵F，计算本征矩阵E
K = np.array(camMat)
height, width, ch = img1.shape
K = np.array([  # for dino
        [2360, 0, width / 2],
        [0, 2360, height / 2],
        [0, 0, 1]])
#数据读Camfixing计算结果即可
# essential matrix
E = np.dot(K.T,F)
E = np.dot(E,K)
print('本征矩阵E\n',E)
################################################################################
#下一步主要是数学知识，还没完全看懂
'''
根据本征矩阵E用SVD分解得到旋转矩阵R和平移向量t，检查旋转矩阵R是否有效，
根据旋转矩阵的标准正交特性判断旋转矩阵的有效性。然后在旋转矩阵有效的情况下构造投影矩阵P0和P1。
'''
W = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
U, S, V = np.linalg.svd(E)

# rotation matrix
R = np.dot(U,W)
R = np.dot(R,V)
print('旋转矩阵R\n',R)
# translation vector
t = [U[0][2], U[1][2], U[2][2]]
print('平移矩阵t\n',t)

#checkValidRot(R)

P1 = [[R[0][0], R[0][1], R[0][2], t[0]], [R[1][0], R[1][1], R[1][2], t[1]], [R[2][0], R[2][1], R[2][2], t[2]]]
P = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]]

################################################################################
 
print('points triangulation')
 
u = []
u1 = []


Kinv = np.linalg.inv(K)
 
# convert points in gray image plane to homogeneous coordinates
for idx in range(len(points1)):
    t = np.dot(Kinv, np.array([points1[idx][0], points1[idx][1], 1.]))
    t1 = np.dot(Kinv, np.array([points2[idx][0], points2[idx][1], 1.]))
    
    u.append(t)   
    u1.append(t1)
################################################################################
 
# re-projection error
reprojError = 0
 
# point cloud (X,Y,Z)
pointCloudX = []
pointCloudY = []
pointCloudZ = []
 
def linearLSTriangulation(u_c, P_c, u_p, P_p):
    """
    Performs linear least squares triangulation via an overdetermined linear
    system
    
    Reference:
    http://users.cecs.anu.edu.au/~hartley/Papers/triangulation/triangulation.pdf
    """    
    u_c=u_c.tolist()
    u_p=u_p.tolist()
    print(u_p[0]*P_p[2][0])
    print(P_p[0][0])
    print(u_p[0]*P_p[2][0] - P_p[0][0])
    A=np.array([[u_c[0]*P_c[2][0] - P_c[0][0], u_c[0]*P_c[2][1] - P_c[0][1], u_c[0]*P_c[2][2] - P_c[0][2]],
              [u_c[1]*P_c[2][0] - P_c[1][0], u_c[1]*P_c[2][1] - P_c[1][1], u_c[1]*P_c[2][2] - P_c[1][2]],
              [u_p[0]*P_p[2][0] - P_p[0][0], u_p[0]*P_p[2][1] - P_p[0][1], u_p[0]*P_p[2][2] - P_p[0][2]],
              [u_p[1]*P_p[2][0] - P_p[1][0], u_p[1]*P_p[2][1] - P_p[1][1], u_p[1]*P_p[2][2] - P_p[1][2]]])

    B=np.array([[-(u_c[0] * P_c[2][3] - P_c[0][3])],
             [-(u_c[1] * P_c[2][3] - P_c[1][3])],
             [-(u_p[0] * P_p[2][3] - P_p[0][3])],
             [-(u_p[1] * P_p[2][3] - P_p[1][3])]])
    # Use of the normal equation, np.linalg.lstsq also works!        
    ret, X = cv.solve(A, B, flags=cv.DECOMP_SVD)
    return X.reshape(3, 1)

#for idx in range(len(points1)):
        
    X = linearLSTriangulation(u[idx], P, u1[idx], P1)
    
    pointCloudX.append(X[0])
    pointCloudY.append(X[1])
    pointCloudZ.append(X[2])
        
    temp = np.zeros(4, np.float32)
    temp[0] = X[0]
    temp[1] = X[1]
    temp[2] = X[2]
    temp[3] = 1.0    
    print(temp)
       
    # calculate re-projection error 
    reprojPoint = np.dot(np.dot(K, P1), temp)
    imgPoint = np.array([points1[idx][0], points1[idx][1], 1.])
    
    reprojError += math.sqrt((reprojPoint[0] / reprojPoint[2] - imgPoint[0]) * (reprojPoint[0] / reprojPoint[2] - imgPoint[0]) + (reprojPoint[1] / reprojPoint[2] - imgPoint[1]) * (reprojPoint[1] / reprojPoint[2] - imgPoint[1]))
P=np.array(P)
P1=np.array(P1)
u=np.array(u)
u1=np.array(u1)
tripoints3d = structure.linear_triangulation(u.T, u1.T, P, P1)
fig = plt.figure()
fig.suptitle('3D reconstructed', fontsize=16)
ax = fig.gca(projection='3d')
ax.plot(tripoints3d[0], tripoints3d[1], tripoints3d[2], 'b.')

#ax.set_zlim(-5,30)	#设置z轴范围
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
ax.view_init(elev=135, azim=90)
plt.show()


#print('Re-project Error:', reprojError / len(points1))


